/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources/extended"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/utils/ptr"
)

const (
	// Name is the name of the plugin used in Registry and configurations.
	Name = names.DynamicResources

	stateKey fwk.StateKey = Name

	// specialClaimInMemName is the name of the special resource claim that
	// exists only in memory. The claim will get a generated name when it is
	// written to API server.
	//
	// It's intentionally not a valid ResourceClaim name to avoid conflicts with
	// some actual ResourceClaim in the apiserver.
	specialClaimInMemName = "<extended-resources>"
	// Field manager used to update the pod status extendedResourceClaimStatus.
	fieldManager = "KubeScheduler"
)

// The state is initialized in PreFilter phase. Because we save the pointer in
// fwk.CycleState, in the later phases we don't need to call Write method
// to update the value
type stateData struct {
	// A copy of all claims for the Pod (i.e. 1:1 match with
	// pod.Spec.ResourceClaims), initially with the status from the start
	// of the scheduling cycle. Each claim instance is read-only because it
	// might come from the informer cache. The instances get replaced when
	// the plugin itself successfully does an Update.
	//
	// When the DRAExtendedResource feature is enabled, the special ResourceClaim
	// which represents extended resource requests is also stored here.
	// The plugin code should treat this field as a black box and only
	// access it via its methods, in particular:
	// - The all method can be used to iterate over all claims, including
	//   the special one.
	// - The allUserClaims method excludes the special one.
	//
	// Empty if the Pod has no claims and no special claim for extended
	// resources backed by DRA, in which case the plugin has no work to do for
	// the Pod.
	claims claimStore

	// draExtendedResource stores data for extended resources backed by DRA.
	// It will remain empty when the DRAExtendedResource feature is disabled.
	draExtendedResource struct {
		// May have extended resource backed by DRA.
		podScalarResources map[v1.ResourceName]int64
		// The mapping of extended resource to device class name
		resourceToDeviceClass map[v1.ResourceName]string
	}

	// Allocator handles claims with structured parameters, which is all of them nowadays.
	allocator structured.Allocator

	// mutex must be locked while accessing any of the fields below.
	mutex sync.Mutex

	// The indices of all claims that:
	// - are allocated
	// - were not available on at least one node
	//
	// Set in parallel during Filter, so write access there must be
	// protected by the mutex. Used by PostFilter.
	unavailableClaims sets.Set[int]

	// informationsForClaim has one entry for each claim in claims.
	informationsForClaim []informationForClaim

	// nodeAllocations caches the result of Filter for the nodes, its key is node name.
	nodeAllocations map[string]nodeAllocation
}

func (d *stateData) Clone() fwk.StateData {
	return d
}

type informationForClaim struct {
	// Node selector based on the claim status if allocated.
	availableOnNodes *nodeaffinity.NodeSelector

	// Set by Reserved, published by PreBind, empty if nothing had to be allocated.
	allocation *resourceapi.AllocationResult
}

// nodeAllocation holds the the allocation results and extended resource claim per node.
type nodeAllocation struct {
	// allocationResults has the allocation results, matching the order of
	// claims which had to be allocated.
	allocationResults []resourceapi.AllocationResult
	// extendedResourceClaim has the special claim for extended resource backed by DRA
	// created during Filter for the nodes.
	extendedResourceClaim *resourceapi.ResourceClaim
}

// draResourceRequest is one entry in a Pods resource request which maps to a DeviceClass.
type draResourceRequest struct {
	count     int64
	className string
}

// DynamicResources is a plugin that ensures that ResourceClaims are allocated.
type DynamicResources struct {
	enabled                    bool
	enableAdminAccess          bool
	enablePrioritizedList      bool
	enableSchedulingQueueHint  bool
	enablePartitionableDevices bool
	enableDeviceTaints         bool
	enableExtendedResource     bool
	enableFilterTimeout        bool
	filterTimeout              time.Duration

	fh         framework.Handle
	clientset  kubernetes.Interface
	celCache   *cel.Cache
	draManager framework.SharedDRAManager
}

// New initializes a new plugin and returns it.
func New(ctx context.Context, plArgs runtime.Object, fh framework.Handle, fts feature.Features) (framework.Plugin, error) {
	if !fts.EnableDynamicResourceAllocation {
		// Disabled, won't do anything.
		return &DynamicResources{}, nil
	}

	args, ok := plArgs.(*config.DynamicResourcesArgs)
	if !ok {
		return nil, fmt.Errorf("got args of type %T, want *DynamicResourcesArgs", plArgs)
	}
	if err := validation.ValidateDynamicResourcesArgs(nil, args, fts); err != nil {
		return nil, err
	}

	pl := &DynamicResources{
		enabled:                    true,
		enableAdminAccess:          fts.EnableDRAAdminAccess,
		enableDeviceTaints:         fts.EnableDRADeviceTaints,
		enablePrioritizedList:      fts.EnableDRAPrioritizedList,
		enableFilterTimeout:        fts.EnableDRASchedulerFilterTimeout,
		enableSchedulingQueueHint:  fts.EnableSchedulingQueueHint,
		enablePartitionableDevices: fts.EnablePartitionableDevices,
		enableExtendedResource:     fts.EnableDRAExtendedResource,
		filterTimeout:              ptr.Deref(args.FilterTimeout, metav1.Duration{}).Duration,

		fh:        fh,
		clientset: fh.ClientSet(),
		// This is a LRU cache for compiled CEL expressions. The most
		// recent 10 of them get reused across different scheduling
		// cycles.
		celCache:   cel.NewCache(10),
		draManager: fh.SharedDRAManager(),
	}

	return pl, nil
}

var _ framework.PreEnqueuePlugin = &DynamicResources{}
var _ framework.PreFilterPlugin = &DynamicResources{}
var _ framework.FilterPlugin = &DynamicResources{}
var _ framework.PostFilterPlugin = &DynamicResources{}
var _ framework.ReservePlugin = &DynamicResources{}
var _ framework.EnqueueExtensions = &DynamicResources{}
var _ framework.PreBindPlugin = &DynamicResources{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *DynamicResources) Name() string {
	return Name
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *DynamicResources) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	if !pl.enabled {
		return nil, nil
	}
	// A resource might depend on node labels for topology filtering.
	// A new or updated node may make pods schedulable.
	//
	// A note about UpdateNodeTaint event:
	// Ideally, it's supposed to register only Add | UpdateNodeLabel because UpdateNodeTaint will never change the result from this plugin.
	// But, we may miss Node/Add event due to preCheck, and we decided to register UpdateNodeTaint | UpdateNodeLabel for all plugins registering Node/Add.
	// See: https://github.com/kubernetes/kubernetes/issues/109437
	nodeActionType := fwk.Add | fwk.UpdateNodeLabel | fwk.UpdateNodeTaint | fwk.UpdateNodeAllocatable
	if pl.enableSchedulingQueueHint {
		// When QHint is enabled, the problematic preCheck is already removed, and we can remove UpdateNodeTaint.
		nodeActionType = fwk.Add | fwk.UpdateNodeLabel | fwk.UpdateNodeAllocatable
	}

	events := []fwk.ClusterEventWithHint{
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: nodeActionType}},
		// Allocation is tracked in ResourceClaims, so any changes may make the pods schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.Add | fwk.Update}, QueueingHintFn: pl.isSchedulableAfterClaimChange},
		// Adding the ResourceClaim name to the pod status makes pods waiting for their ResourceClaim schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdatePodGeneratedResourceClaim}, QueueingHintFn: pl.isSchedulableAfterPodChange},
		// A pod might be waiting for a class to get created or modified.
		{Event: fwk.ClusterEvent{Resource: fwk.DeviceClass, ActionType: fwk.Add | fwk.Update}},
		// Adding or updating a ResourceSlice might make a pod schedulable because new resources became available.
		{Event: fwk.ClusterEvent{Resource: fwk.ResourceSlice, ActionType: fwk.Add | fwk.Update}},
	}

	return events, nil
}

// PreEnqueue checks if there are known reasons why a pod currently cannot be
// scheduled. When this fails, one of the registered events can trigger another
// attempt.
func (pl *DynamicResources) PreEnqueue(ctx context.Context, pod *v1.Pod) (status *fwk.Status) {
	if !pl.enabled {
		return nil
	}

	if err := pl.foreachPodResourceClaim(pod, nil); err != nil {
		return statusUnschedulable(klog.FromContext(ctx), err.Error())
	}
	return nil
}

// isSchedulableAfterClaimChange is invoked for add and update claim events reported by
// an informer. It checks whether that change made a previously unschedulable
// pod schedulable. It errs on the side of letting a pod scheduling attempt
// happen. The delete claim event will not invoke it, so newObj will never be nil.
func (pl *DynamicResources) isSchedulableAfterClaimChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalClaim, modifiedClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
	if err != nil {
		// Shouldn't happen.
		return fwk.Queue, fmt.Errorf("unexpected object in isSchedulableAfterClaimChange: %w", err)
	}

	usesClaim := false
	if err := pl.foreachPodResourceClaim(pod, func(_ string, claim *resourceapi.ResourceClaim) {
		if claim.UID == modifiedClaim.UID {
			usesClaim = true
		}
	}); err != nil {
		// This is not an unexpected error: we know that
		// foreachPodResourceClaim only returns errors for "not
		// schedulable".
		if loggerV := logger.V(6); loggerV.Enabled() {
			owner := metav1.GetControllerOf(modifiedClaim)
			loggerV.Info("pod is not schedulable after resource claim change", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim), "claimOwner", owner, "reason", err.Error())
		}
		return fwk.QueueSkip, nil
	}

	if originalClaim != nil &&
		originalClaim.Status.Allocation != nil &&
		modifiedClaim.Status.Allocation == nil {
		// A claim with structured parameters was deallocated. This might have made
		// resources available for other pods.
		logger.V(6).Info("claim with structured parameters got deallocated", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		return fwk.Queue, nil
	}

	if !usesClaim {
		// This was not the claim the pod was waiting for.
		logger.V(6).Info("unrelated claim got modified", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		return fwk.QueueSkip, nil
	}

	if originalClaim == nil {
		logger.V(5).Info("claim for pod got created", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		return fwk.Queue, nil
	}

	// Modifications may or may not be relevant. If the entire
	// status is as before, then something else must have changed
	// and we don't care. What happens in practice is that the
	// resource driver adds the finalizer.
	if apiequality.Semantic.DeepEqual(&originalClaim.Status, &modifiedClaim.Status) {
		if loggerV := logger.V(7); loggerV.Enabled() {
			// Log more information.
			loggerV.Info("claim for pod got modified where the pod doesn't care", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim), "diff", diff.Diff(originalClaim, modifiedClaim))
		} else {
			logger.V(6).Info("claim for pod got modified where the pod doesn't care", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		}
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("status of claim for pod got updated", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
	return fwk.Queue, nil
}

// isSchedulableAfterPodChange is invoked for update pod events reported by
// an informer. It checks whether that change adds the ResourceClaim(s) that the
// pod has been waiting for.
func (pl *DynamicResources) isSchedulableAfterPodChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, modifiedPod, err := schedutil.As[*v1.Pod](nil, newObj)
	if err != nil {
		// Shouldn't happen.
		return fwk.Queue, fmt.Errorf("unexpected object in isSchedulableAfterClaimChange: %w", err)
	}

	if pod.UID != modifiedPod.UID {
		logger.V(7).Info("pod is not schedulable after change in other pod", "pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		return fwk.QueueSkip, nil
	}

	if err := pl.foreachPodResourceClaim(modifiedPod, nil); err != nil {
		// This is not an unexpected error: we know that
		// foreachPodResourceClaim only returns errors for "not
		// schedulable".
		logger.V(6).Info("pod is not schedulable after being updated", "pod", klog.KObj(pod))
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("pod got updated and is schedulable", "pod", klog.KObj(pod))
	return fwk.Queue, nil
}

// podResourceClaims returns the ResourceClaims for all pod.Spec.PodResourceClaims.
func (pl *DynamicResources) podResourceClaims(pod *v1.Pod) ([]*resourceapi.ResourceClaim, error) {
	claims := make([]*resourceapi.ResourceClaim, 0, len(pod.Spec.ResourceClaims))
	if err := pl.foreachPodResourceClaim(pod, func(_ string, claim *resourceapi.ResourceClaim) {
		// We store the pointer as returned by the lister. The
		// assumption is that if a claim gets modified while our code
		// runs, the cache will store a new pointer, not mutate the
		// existing object that we point to here.
		claims = append(claims, claim)
	}); err != nil {
		return nil, err
	}
	return claims, nil
}

// foreachPodResourceClaim checks that each ResourceClaim for the pod exists.
// It calls an optional handler for those claims that it finds.
func (pl *DynamicResources) foreachPodResourceClaim(pod *v1.Pod, cb func(podResourceName string, claim *resourceapi.ResourceClaim)) error {
	for _, resource := range pod.Spec.ResourceClaims {
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, &resource)
		if err != nil {
			return err
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}
		claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
		if err != nil {
			return err
		}

		if claim.DeletionTimestamp != nil {
			return fmt.Errorf("resourceclaim %q is being deleted", claim.Name)
		}

		if mustCheckOwner {
			if err := resourceclaim.IsForPod(pod, claim); err != nil {
				return err
			}
		}
		if cb != nil {
			cb(resource.Name, claim)
		}
	}
	return nil
}

// hasDeviceClassMappedExtendedResource returns true when the given resource list has an extended resource, that has
// a mapping to a device class.
func hasDeviceClassMappedExtendedResource(reqs v1.ResourceList, deviceClassMapping map[v1.ResourceName]string) bool {
	for rName, rValue := range reqs {
		if rValue.IsZero() {
			// We only care about the resources requested by the pod we are trying to schedule.
			continue
		}
		switch rName {
		case v1.ResourceCPU:
		case v1.ResourceMemory:
		case v1.ResourceEphemeralStorage:
		default:
			if v1helper.IsExtendedResourceName(rName) {
				_, ok := deviceClassMapping[rName]
				if ok {
					return true
				}
			}
		}
	}
	return false
}

// findExtendedResourceClaim looks for the extended resource claim, i.e., the claim with special annotation
// set to "true", and with the pod as owner. It must be called with all ResourceClaims in the cluster.
// The returned ResourceClaim is read-only.
func findExtendedResourceClaim(pod *v1.Pod, resourceClaims []*resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	// TODO: check performance of iterating over all claims and consider indexing claims by pod.
	// TODO: use pod status as shortcut?
	for _, c := range resourceClaims {
		if c.Annotations[resourceapi.ExtendedResourceClaimAnnotation] == "true" {
			for _, or := range c.OwnerReferences {
				if or.Name == pod.Name && *or.Controller && or.UID == pod.UID {
					return c
				}
			}
		}
	}
	return nil
}

// preFilterExtendedResources checks if there is any extended resource in the
// pod requests that has a device class mapping, i.e., there is a device class
// that has spec.ExtendedResourceName or its implicit extended resource name
// matching the given extended resource in that pod requests.
//
// It looks for the special resource claim for the pod created from prior scheduling
// cycle. If not found, it creates the special claim with  no Requests in the Spec,
// with a temporary UID, and the specialClaimInMemName name.
// Either way, the special claim is stored in state.claims.
//
// In addition following fields are also stored in cycle state
// 1. resourceToDeviceClassMapping
// 2. podScalarResources
//
// It returns the special ResourceClaim and an error status. It returns nil for both
// if the feature is disabled or not required for the Pod.
func (pl *DynamicResources) preFilterExtendedResources(pod *v1.Pod, logger klog.Logger, s *stateData, claims []*resourceapi.ResourceClaim) (*resourceapi.ResourceClaim, *fwk.Status) {
	if !pl.enableExtendedResource {
		return nil, nil
	}

	deviceClassMapping, err := extended.DeviceclassMapping(pl.draManager)
	if err != nil {
		return nil, statusUnschedulable(logger, err.Error())
	}

	reqs := resourcehelper.PodRequests(pod, resourcehelper.PodResourcesOptions{})
	hasExtendedResource := hasDeviceClassMappedExtendedResource(reqs, deviceClassMapping)
	if !hasExtendedResource {
		return nil, nil
	}

	s.draExtendedResource.resourceToDeviceClass = deviceClassMapping
	r := framework.NewResource(reqs)
	s.draExtendedResource.podScalarResources = r.ScalarResources

	resourceClaims, err := pl.draManager.ResourceClaims().List()
	if err != nil {
		return nil, statusUnschedulable(logger, err.Error(), "listing ResourceClaims")
	}

	// Check if the special resource claim has been created from prior scheduling cycle.
	//
	// If it was already allocated earlier, that allocation might not be valid anymore.
	// We could try to check that, but it depends on various factors that are difficult to
	// cover (basically needs to replicate allocator logic) and if it turns out that the
	// allocation is stale, we would have to schedule with those allocated devices not
	// available for a new allocation. This situation should be rare (= binding failure),
	// so we solve it via brute-force (TODO, not implemented yet!):
	// - Kick off deallocation in the background.
	// - Mark the pod as unschedulable. Successful deallocation will make it schedulable again.
	extendedResourceClaim := findExtendedResourceClaim(pod, resourceClaims)
	if extendedResourceClaim == nil {
		// Create one special claim for all extended resources backed by DRA in the Pod.
		// Create the ResourceClaim with pod as owner, with a generated name that uses
		// <pod name>-extended-resources- as base. The final name will get truncated if it
		// would be too long.
		extendedResourceClaim = &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: pod.Namespace,
				Name:      specialClaimInMemName,
				// fake temporary UID for use in SignalClaimPendingAllocation
				UID:          types.UID(uuid.NewUUID()),
				GenerateName: pod.Name + "-extended-resources-",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion:         "v1",
						Kind:               "Pod",
						Name:               pod.Name,
						UID:                pod.UID,
						Controller:         ptr.To(true),
						BlockOwnerDeletion: ptr.To(true),
					},
				},
				Annotations: map[string]string{
					resourceapi.ExtendedResourceClaimAnnotation: "true",
				},
			},
			Spec: resourceapi.ResourceClaimSpec{},
		}
	}
	return extendedResourceClaim, nil
}

// PreFilter invoked at the prefilter extension point to check if pod has all
// immediate claims bound. UnschedulableAndUnresolvable is returned if
// the pod cannot be scheduled at the moment on any node.
func (pl *DynamicResources) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []*framework.NodeInfo) (res *framework.PreFilterResult, status *fwk.Status) {
	if !pl.enabled {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	logger := klog.FromContext(ctx)

	// If the pod does not reference any claim, we don't need to do
	// anything for it. We just initialize an empty state to record that
	// observation for the other functions. This gets updated below
	// if we get that far.
	s := &stateData{}
	state.Write(stateKey, s)

	userClaims, err := pl.podResourceClaims(pod)
	if err != nil {
		return nil, statusUnschedulable(logger, err.Error())
	}
	logger.V(5).Info("pod resource claims", "pod", klog.KObj(pod), "resourceclaims", klog.KObjSlice(userClaims))
	extendedResourceClaim, status := pl.preFilterExtendedResources(pod, logger, s, userClaims)
	if status != nil {
		return nil, status
	}
	claims := newClaimStore(userClaims, extendedResourceClaim)

	// This check covers user and extended ResourceClaim.
	if claims.empty() {
		return nil, fwk.NewStatus(fwk.Skip)
	}

	// Counts all claims which the scheduler needs to allocate itself.
	numClaimsToAllocate := 0
	s.informationsForClaim = make([]informationForClaim, claims.len())
	for index, claim := range claims.all() {
		if claim.Status.Allocation != nil &&
			!resourceclaim.CanBeReserved(claim) &&
			!resourceclaim.IsReservedForPod(pod, claim) {
			// Resource is in use. The pod has to wait.
			return nil, statusUnschedulable(logger, "resourceclaim in use", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
		}

		if claim.Status.Allocation != nil {
			if claim.Status.Allocation.NodeSelector != nil {
				nodeSelector, err := nodeaffinity.NewNodeSelector(claim.Status.Allocation.NodeSelector)
				if err != nil {
					return nil, statusError(logger, err)
				}
				s.informationsForClaim[index].availableOnNodes = nodeSelector
			}
		} else {
			numClaimsToAllocate++

			// Allocation in flight? Better wait for that
			// to finish, see inFlightAllocations
			// documentation for details.
			if pl.draManager.ResourceClaims().ClaimHasPendingAllocation(claim.UID) {
				return nil, statusUnschedulable(logger, fmt.Sprintf("resource claim %s is in the process of being allocated", klog.KObj(claim)))
			}

			// Continue without validating the special claim for extended resource backed by DRA.
			// For the claim template, it is not allocated yet at this point, and it does not have a spec.
			// For the claim from prior scheduling cycle, leave it to Filter Phase to validate.
			if claim.Name == specialClaimInMemName {
				continue
			}

			// Check all requests and device classes. If a class
			// does not exist, scheduling cannot proceed, no matter
			// how the claim is being allocated.
			//
			// When using a control plane controller, a class might
			// have a node filter. This is useful for trimming the
			// initial set of potential nodes before we ask the
			// driver(s) for information about the specific pod.
			for _, request := range claim.Spec.Devices.Requests {
				// The requirements differ depending on whether the request has a list of
				// alternative subrequests defined in the firstAvailable field.
				if len(request.FirstAvailable) == 0 {
					if status := pl.validateDeviceClass(logger, request.DeviceClassName, request.Name); status != nil {
						return nil, status
					}
				} else {
					if !pl.enablePrioritizedList {
						return nil, statusUnschedulable(logger, fmt.Sprintf("resource claim %s, request %s: has subrequests, but the DRAPrioritizedList feature is disabled", klog.KObj(claim), request.Name))
					}
					for _, subRequest := range request.FirstAvailable {
						qualRequestName := strings.Join([]string{request.Name, subRequest.Name}, "/")
						if status := pl.validateDeviceClass(logger, subRequest.DeviceClassName, qualRequestName); status != nil {
							return nil, status
						}
					}
				}
			}
		}
	}

	if numClaimsToAllocate > 0 {
		if loggerV := logger.V(5); loggerV.Enabled() {
			claimsToAllocate := make([]*resourceapi.ResourceClaim, 0, claims.len())
			for _, claim := range claims.toAllocate() {
				claimsToAllocate = append(claimsToAllocate, claim)
			}
			loggerV.Info("Preparing allocation with structured parameters", "pod", klog.KObj(pod), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		}

		// Doing this over and over again for each pod could be avoided
		// by setting the allocator up once and then keeping it up-to-date
		// as changes are observed.
		//
		// But that would cause problems for using the plugin in the
		// Cluster Autoscaler. If this step here turns out to be
		// expensive, we may have to maintain and update state more
		// persistently.
		//
		// Claims (and thus their devices) are treated as "allocated" if they are in the assume cache
		// or currently their allocation is in-flight. This does not change
		// during filtering, so we can determine that once.
		allAllocatedDevices, err := pl.draManager.ResourceClaims().ListAllAllocatedDevices()
		if err != nil {
			return nil, statusError(logger, err)
		}
		slices, err := pl.draManager.ResourceSlices().ListWithDeviceTaintRules()
		if err != nil {
			return nil, statusError(logger, err)
		}
		features := structured.Features{
			AdminAccess:          pl.enableAdminAccess,
			PrioritizedList:      pl.enablePrioritizedList,
			PartitionableDevices: pl.enablePartitionableDevices,
			DeviceTaints:         pl.enableDeviceTaints,
		}
		allocator, err := structured.NewAllocator(ctx, features, allAllocatedDevices, pl.draManager.DeviceClasses(), slices, pl.celCache)
		if err != nil {
			return nil, statusError(logger, err)
		}
		s.allocator = allocator
		s.nodeAllocations = make(map[string]nodeAllocation)
	}
	s.claims = claims
	return nil, nil
}

func (pl *DynamicResources) validateDeviceClass(logger klog.Logger, deviceClassName, requestName string) *fwk.Status {
	if deviceClassName == "" {
		return statusError(logger, fmt.Errorf("request %s: unsupported request type", requestName))
	}

	_, err := pl.draManager.DeviceClasses().Get(deviceClassName)
	if err != nil {
		// If the class cannot be retrieved, allocation cannot proceed.
		if apierrors.IsNotFound(err) {
			// Here we mark the pod as "unschedulable", so it'll sleep in
			// the unscheduleable queue until a DeviceClass event occurs.
			return statusUnschedulable(logger, fmt.Sprintf("request %s: device class %s does not exist", requestName, deviceClassName))
		}
	}
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *DynamicResources) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func getStateData(cs fwk.CycleState) (*stateData, error) {
	state, err := cs.Read(stateKey)
	if err != nil {
		return nil, err
	}
	s, ok := state.(*stateData)
	if !ok {
		return nil, errors.New("unable to convert state into stateData")
	}
	return s, nil
}

// filterExtendedResources computes the special claim's Requests based on the
// node's Allocatable. It returns the special claim updated to match what needs
// to be allocated through DRA for the node or nil if nothing needs to be allocated.
//
// It returns an error when the pod's extended resource requests cannot be allocated
// from node's Allocatable, nor matching any device class's explicit or implicit
// ExtendedResourceName.
func (pl *DynamicResources) filterExtendedResources(state *stateData, pod *v1.Pod, nodeInfo *framework.NodeInfo, logger klog.Logger) (*resourceapi.ResourceClaim, *fwk.Status) {
	extendedResourceClaim := state.claims.extendedResourceClaim()
	if extendedResourceClaim == nil {
		// Nothing to do.
		return nil, nil
	}

	extendedResources := make(map[v1.ResourceName]int64)
	hasExtendedResource := false
	for rName, rQuant := range state.draExtendedResource.podScalarResources {
		if !v1helper.IsExtendedResourceName(rName) {
			continue
		}
		// Skip in case request quantity is zero
		if rQuant == 0 {
			continue
		}

		_, okScalar := nodeInfo.Allocatable.ScalarResources[rName]
		_, okDynamic := state.draExtendedResource.resourceToDeviceClass[rName]
		if okDynamic {
			if okScalar {
				// node provides the resource via device plugin
				extendedResources[rName] = 0
			} else {
				// node needs to provide the resource via DRA
				extendedResources[rName] = rQuant
				hasExtendedResource = true
			}
		} else if !okScalar {
			// has request neither provided by device plugin, nor backed by DRA,
			// hence the pod does not fit the node.
			return nil, statusUnschedulable(logger, "cannot fit resource", "pod", klog.KObj(pod), "node", klog.KObj(nodeInfo.Node()), "resource", rName)
		}
	}

	// No extended resources backed by DRA on this node.
	// The pod may have extended resources, but they are all backed by device
	// plugin, hence the noderesources plugin should have checked if the node
	// can fit the pod.
	// This dynamic resources plugin Filter phase has nothing left to do.
	if state.claims.len() == 1 && !hasExtendedResource {
		// TODO: what if the claim is allocated?
		return nil, nil
	}

	// Each node needs its own, potentially different variant of the claim.
	nodeExtendedResourceClaim := extendedResourceClaim.DeepCopy()

	// Creating the extended resource claim's Requests by
	// iterating over the containers, and the resources in the containers,
	// and create one request per <container, extended resource>.
	containers := slices.Clone(pod.Spec.InitContainers)
	containers = append(containers, pod.Spec.Containers...)
	// Creating the extended resource claim's Requests by
	for r := range extendedResources {
		for i, c := range containers {
			creqs := c.Resources.Requests
			if creqs == nil {
				continue
			}
			var rQuant resource.Quantity
			var ok bool
			if rQuant, ok = creqs[r]; !ok {
				continue
			}
			crq, ok := (&rQuant).AsInt64()
			if !ok || crq == 0 {
				continue
			}
			className, ok := state.draExtendedResource.resourceToDeviceClass[r]
			// skip if the request does not map to a device class
			if !ok || className == "" {
				continue
			}
			keys := make([]string, 0, len(creqs))
			for k := range creqs {
				keys = append(keys, k.String())
			}
			slice.SortStrings(keys)
			ridx := 0
			for j := range keys {
				if keys[j] == r.String() {
					ridx = j
					break
				}
			}
			// i is the index of the container if the list of initContainers + containers.
			// ridx is the index of the extended resource request in the sorted all requests in the container.
			// crq is the quantity of the extended resource request.
			nodeExtendedResourceClaim.Spec.Devices.Requests = append(nodeExtendedResourceClaim.Spec.Devices.Requests,
				resourceapi.DeviceRequest{
					Name:            fmt.Sprintf("container-%d-request-%d", i, ridx), // need to be container name index - external resource name index
					DeviceClassName: className,                                       // map external resource name -> device class name
					AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
					Count:           crq,
				})
		}
	}

	if extendedResourceClaim.Status.Allocation != nil {
		// If it is already allocated, then we cannot simply allocate it again.
		// See TODO in preFilterExtendedResources.
		//
		// Perhaps we can do a semantic compare here to determine whether the
		// old and new requests are equivalent?
		return nil, nil
	}

	return nodeExtendedResourceClaim, nil
}

// Filter invoked at the filter extension point.
// It evaluates if a pod can fit due to the resources it requests,
// for both allocated and unallocated claims.
//
// For claims that are bound, then it checks that the node affinity is
// satisfied by the given node.
//
// For claims that are unbound, it checks whether the claim might get allocated
// for the node.
func (pl *DynamicResources) Filter(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *fwk.Status {
	if !pl.enabled {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return nil
	}

	logger := klog.FromContext(ctx)
	node := nodeInfo.Node()
	nodeExtendedResourceClaim, status := pl.filterExtendedResources(state, pod, nodeInfo, logger)
	if status != nil {
		return status
	}

	var unavailableClaims []int
	for index, claim := range state.claims.all() {
		logger.V(10).Info("filtering based on resource claims of the pod", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))

		// This node selector only gets set if the claim is allocated.
		if nodeSelector := state.informationsForClaim[index].availableOnNodes; nodeSelector != nil && !nodeSelector.Match(node) {
			logger.V(5).Info("allocation's node selector does not match", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))
			unavailableClaims = append(unavailableClaims, index)
		}
	}

	// Use allocator to check the node and cache the result in case that the node is picked.
	var allocations []resourceapi.AllocationResult
	if state.allocator != nil {
		allocCtx := ctx
		if loggerV := logger.V(5); loggerV.Enabled() {
			allocCtx = klog.NewContext(allocCtx, klog.LoggerWithValues(logger, "node", klog.KObj(node)))
		}

		// Apply timeout to the operation?
		if pl.enableFilterTimeout && pl.filterTimeout > 0 {
			c, cancel := context.WithTimeout(allocCtx, pl.filterTimeout)
			defer cancel()
			allocCtx = c
		}

		// Check which claims need to be allocated.
		//
		// This replaces the special ResourceClaim for extended resources with one
		// matching the node.
		claimsToAllocate := make([]*resourceapi.ResourceClaim, 0, state.claims.len())
		extendedResourceClaim := state.claims.extendedResourceClaim()
		for _, claim := range state.claims.toAllocate() {
			if claim == extendedResourceClaim {
				claim = nodeExtendedResourceClaim
			}
			claimsToAllocate = append(claimsToAllocate, claim)
		}
		a, err := state.allocator.Allocate(allocCtx, node, claimsToAllocate)
		switch {
		case errors.Is(err, context.DeadlineExceeded):
			return statusUnschedulable(logger, "timed out trying to allocate devices", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		case ctx.Err() != nil:
			return statusUnschedulable(logger, fmt.Sprintf("asked by caller to stop allocating devices: %v", context.Cause(ctx)), "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		case err != nil:
			// This should only fail if there is something wrong with the claim or class.
			// Return an error to abort scheduling of it.
			//
			// This will cause retries. It would be slightly nicer to mark it as unschedulable
			// *and* abort scheduling. Then only cluster event for updating the claim or class
			// with the broken CEL expression would trigger rescheduling.
			//
			// But we cannot do both. As this shouldn't occur often, aborting like this is
			// better than the more complicated alternative (return Unschedulable here, remember
			// the error, then later raise it again later if needed).
			return statusError(logger, err, "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		}
		// Check for exact length just to be sure. In practice this is all-or-nothing.
		if len(a) != len(claimsToAllocate) {
			return statusUnschedulable(logger, "cannot allocate all claims", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		}
		// Reserve uses this information.
		allocations = a
	}

	// Store information in state while holding the mutex.
	if state.allocator != nil || len(unavailableClaims) > 0 {
		state.mutex.Lock()
		defer state.mutex.Unlock()
	}

	if len(unavailableClaims) > 0 {
		// Remember all unavailable claims. This might be observed
		// concurrently, so we have to lock the state before writing.

		if state.unavailableClaims == nil {
			state.unavailableClaims = sets.New[int]()
		}

		for _, index := range unavailableClaims {
			state.unavailableClaims.Insert(index)
		}
		return statusUnschedulable(logger, "resourceclaim not available on the node", "pod", klog.KObj(pod))
	}

	if state.allocator != nil {
		state.nodeAllocations[node.Name] = nodeAllocation{
			allocationResults:     allocations,
			extendedResourceClaim: nodeExtendedResourceClaim,
		}
	}

	return nil
}

// PostFilter checks whether there are allocated claims that could get
// deallocated to help get the Pod schedulable. If yes, it picks one and
// requests its deallocation.  This only gets called when filtering found no
// suitable node.
func (pl *DynamicResources) PostFilter(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap framework.NodeToStatusReader) (*framework.PostFilterResult, *fwk.Status) {
	if !pl.enabled {
		return nil, fwk.NewStatus(fwk.Unschedulable, "plugin disabled")
	}

	logger := klog.FromContext(ctx)
	state, err := getStateData(cs)
	if err != nil {
		return nil, statusError(logger, err)
	}
	// If a Pod doesn't have any resource claims attached to it, there is no need for further processing.
	// Thus we provide a fast path for this case to avoid unnecessary computations.
	if state.claims.empty() {
		return nil, fwk.NewStatus(fwk.Unschedulable, "no new claims to deallocate")
	}
	extendedResourceClaim := state.claims.extendedResourceClaim()

	// Iterating over a map is random. This is intentional here, we want to
	// pick one claim randomly because there is no better heuristic.
	for index := range state.unavailableClaims {
		claim := state.claims.get(index)
		if claim == extendedResourceClaim {
			// Handled below.
			continue
		}

		if len(claim.Status.ReservedFor) == 0 ||
			len(claim.Status.ReservedFor) == 1 && claim.Status.ReservedFor[0].UID == pod.UID {
			claim := claim.DeepCopy()
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
			logger.V(5).Info("Deallocation of ResourceClaim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
			if _, err := pl.clientset.ResourceV1beta1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{}); err != nil {
				return nil, statusError(logger, err)
			}
			return nil, fwk.NewStatus(fwk.Unschedulable, "deallocation of ResourceClaim completed")
		}
	}

	if extendedResourceClaim != nil && extendedResourceClaim.Name != specialClaimInMemName {
		// If the special resource claim for extended resource backed by DRA
		// is reserved or allocated at prior scheduling cycle, then it should be deleted.
		extendedResourceClaim := extendedResourceClaim.DeepCopy()

		// Remove the finalizer to unblock removal first.
		builtinControllerFinalizer := slices.Index(extendedResourceClaim.Finalizers, resourceapi.Finalizer)
		if builtinControllerFinalizer >= 0 {
			extendedResourceClaim.Finalizers = slices.Delete(extendedResourceClaim.Finalizers, builtinControllerFinalizer, builtinControllerFinalizer+1)
			retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
				_, err := pl.clientset.ResourceV1beta1().ResourceClaims(extendedResourceClaim.Namespace).Update(ctx, extendedResourceClaim, metav1.UpdateOptions{})
				if err != nil {
					return fmt.Errorf("update resourceclaim %s/%s: %w", extendedResourceClaim.Namespace, extendedResourceClaim.Name, err)
				}
				return nil
			})
			if retryErr != nil {
				return nil, statusError(logger, retryErr)
			}
		}

		// TODO: if the claim is allocated or reserved, that must be cleared before it can be deleted. Need integration-level tests for this,
		// the fake client doesn't enforce that semantic.

		// Delete the special claim created by scheduler for the extended resource backed by DRA in prior scheduling cycle.
		logger.V(5).Info("Delete ResourceClaim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(extendedResourceClaim))
		err := pl.clientset.ResourceV1beta1().ResourceClaims(extendedResourceClaim.Namespace).Delete(ctx, extendedResourceClaim.Name, metav1.DeleteOptions{})
		if err != nil {
			return nil, statusError(logger, err)
		}
	}

	return nil, fwk.NewStatus(fwk.Unschedulable, "still not schedulable")
}

// Reserve reserves claims for the pod.
func (pl *DynamicResources) Reserve(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) (status *fwk.Status) {
	if !pl.enabled {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return nil
	}

	logger := klog.FromContext(ctx)

	numClaimsWithAllocator := 0
	for _, claim := range state.claims.all() {
		if claim.Status.Allocation != nil {
			// Allocated, but perhaps not reserved yet. We checked in PreFilter that
			// the pod could reserve the claim. Instead of reserving here by
			// updating the ResourceClaim status, we assume that reserving
			// will work and only do it for real during binding. If it fails at
			// that time, some other pod was faster and we have to try again.
			continue
		}

		numClaimsWithAllocator++
	}

	if numClaimsWithAllocator == 0 {
		// Nothing left to do.
		return nil
	}
	extendedResourceClaim := state.claims.extendedResourceClaim()
	numClaimsToAllocate := 0
	needToAllocateUserClaims := false
	for _, claim := range state.claims.toAllocate() {
		numClaimsToAllocate++
		if claim != extendedResourceClaim {
			needToAllocateUserClaims = true
		}
	}

	// Prepare allocation of claims handled by the schedulder.
	if state.allocator != nil {
		// Entries in these two slices match each other.
		allocations, ok := state.nodeAllocations[nodeName]
		if !ok || len(allocations.allocationResults) == 0 {
			// This can happen only when claimsToAllocate has a single special claim template for extended resource backed by DRA,
			// But it is satisfied by the node with device plugin, hence no DRA allocation.
			if !needToAllocateUserClaims {
				return nil
			}

			// We checked before that the node is suitable. This shouldn't have failed,
			// so treat this as an error.
			return statusError(logger, errors.New("claim allocation not found for node"))
		}

		// Sanity check: do we have results for all pending claims?
		if len(allocations.allocationResults) != numClaimsToAllocate ||
			len(allocations.allocationResults) != numClaimsWithAllocator {
			return statusError(logger, fmt.Errorf("internal error, have %d allocations, %d claims to allocate, want %d claims", len(allocations.allocationResults), numClaimsToAllocate, numClaimsWithAllocator))
		}

		for i, claim := range state.claims.toAllocate() {
			index := state.claims.index(claim)
			if index < 0 {
				return statusError(logger, fmt.Errorf("internal error, claim %s with allocation not found", claim.Name))
			}
			allocation := &allocations.allocationResults[i]
			state.informationsForClaim[index].allocation = allocation

			if claim == extendedResourceClaim {
				// replace the special claim template for extended
				// resource backed by DRA with the real instantiated claim.
				claim = allocations.extendedResourceClaim
			}

			// Strictly speaking, we don't need to store the full modified object.
			// The allocation would be enough. The full object is useful for
			// debugging, testing and the allocator, so let's make it realistic.
			claim = claim.DeepCopy()
			if !slices.Contains(claim.Finalizers, resourceapi.Finalizer) {
				claim.Finalizers = append(claim.Finalizers, resourceapi.Finalizer)
			}
			claim.Status.Allocation = allocation
			err := pl.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim)
			if err != nil {
				return statusError(logger, fmt.Errorf("internal error, couldn't signal allocation for claim %s", claim.Name))
			}
			logger.V(5).Info("Reserved resource in allocation result", "claim", klog.KObj(claim), "allocation", klog.Format(allocation))
		}
	}

	return nil
}

// Unreserve clears the ReservedFor field for all claims.
// It's idempotent, and does nothing if no state found for the given pod.
func (pl *DynamicResources) Unreserve(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) {
	if !pl.enabled {
		return
	}
	state, err := getStateData(cs)
	if err != nil {
		return
	}
	if state.claims.empty() {
		return
	}

	logger := klog.FromContext(ctx)

	for _, claim := range state.claims.allUserClaims() {
		// If allocation was in-flight, then it's not anymore and we need to revert the
		// claim object in the assume cache to what it was before.
		if deleted := pl.draManager.ResourceClaims().RemoveClaimPendingAllocation(claim.UID); deleted {
			pl.draManager.ResourceClaims().AssumedClaimRestore(claim.Namespace, claim.Name)
		}

		if claim.Status.Allocation != nil &&
			resourceclaim.IsReservedForPod(pod, claim) {
			// Remove pod from ReservedFor. A strategic-merge-patch is used
			// because that allows removing an individual entry without having
			// the latest slice.
			patch := fmt.Sprintf(`{"metadata": {"uid": %q}, "status": { "reservedFor": [ {"$patch": "delete", "uid": %q} ] }}`,
				claim.UID,
				pod.UID,
			)
			logger.V(5).Info("unreserve", "resourceclaim", klog.KObj(claim), "pod", klog.KObj(pod))
			claim, err := pl.clientset.ResourceV1beta1().ResourceClaims(claim.Namespace).Patch(ctx, claim.Name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}, "status")
			if err != nil {
				// We will get here again when pod scheduling is retried.
				logger.Error(err, "unreserve", "resourceclaim", klog.KObj(claim))
			}
		}
	}

	extendedResourceClaim := state.claims.extendedResourceClaim()
	if extendedResourceClaim == nil {
		return
	}

	pl.draManager.ResourceClaims().RemoveClaimPendingAllocation(extendedResourceClaim.UID)

	// No matter RemoveClaimPendingAllocation return 'deleted' or not,
	// Assume cache needs to be restored when it is the special claim backed by DRA.
	// TODO: why?
	pl.draManager.ResourceClaims().AssumedClaimRestore(extendedResourceClaim.Namespace, extendedResourceClaim.Name)
	if extendedResourceClaim.Name == specialClaimInMemName {
		return
	}
	logger.V(5).Info("delete extended resource backed by DRA", "resourceclaim", klog.KObj(extendedResourceClaim), "pod", klog.KObj(pod), "claim.UID", extendedResourceClaim.UID)

	// Remove the finalizer to unblock deletion.
	extendedResourceClaim, err = pl.clientset.ResourceV1beta1().ResourceClaims(extendedResourceClaim.Namespace).Get(ctx, extendedResourceClaim.Name, metav1.GetOptions{})
	if err != nil {
		// We will get here again when pod scheduling is retried.
		logger.Error(err, "get", "resourceclaim", klog.KObj(extendedResourceClaim))
		return
	}
	builtinControllerFinalizer := slices.Index(extendedResourceClaim.Finalizers, resourceapi.Finalizer)
	if builtinControllerFinalizer >= 0 {
		extendedResourceClaim.Finalizers = slices.Delete(extendedResourceClaim.Finalizers, builtinControllerFinalizer, builtinControllerFinalizer+1)
		retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			_, err := pl.clientset.ResourceV1beta1().ResourceClaims(extendedResourceClaim.Namespace).Update(ctx, extendedResourceClaim, metav1.UpdateOptions{})
			if err != nil {
				return fmt.Errorf("update resourceclaim %s/%s: %w", extendedResourceClaim.Namespace, extendedResourceClaim.Name, err)
			}
			return nil
		})
		if retryErr != nil {
			logger.Error(err, "update", "resourceclaim", klog.KObj(extendedResourceClaim))
		}
	}

	// TODO: if the claim is allocated or reserved, that must be cleared before it can be deleted. Need integration-level tests for this,
	// the fake client doesn't enforce that semantic.

	// Delete the claim created by scheduler for the extended resource backed by DRA
	err = pl.clientset.ResourceV1beta1().ResourceClaims(extendedResourceClaim.Namespace).Delete(ctx, extendedResourceClaim.Name, metav1.DeleteOptions{})
	if err != nil {
		// We will get here again when pod scheduling is retried.
		logger.Error(err, "delete", "resourceclaim", klog.KObj(extendedResourceClaim))
	}
}

// PreBind gets called in a separate goroutine after it has been determined
// that the pod should get bound to this node. Because Reserve did not actually
// reserve claims, we need to do it now. For claims with the builtin controller,
// we also handle the allocation.
//
// If anything fails, we return an error and
// the pod will have to go into the backoff queue. The scheduler will call
// Unreserve as part of the error handling.
func (pl *DynamicResources) PreBind(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if !pl.enabled {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return nil
	}

	logger := klog.FromContext(ctx)

	for index, claim := range state.claims.all() {
		if !resourceclaim.IsReservedForPod(pod, claim) {
			claim, err := pl.bindClaim(ctx, state, index, pod, nodeName)
			if err != nil {
				return statusError(logger, err)
			}
			// Updated here such that Unreserve can work with patched claim.
			state.claims.set(index, claim)
		}
	}
	// If we get here, we know that reserving the claim for
	// the pod worked and we can proceed with binding it.
	return nil
}

// PreBindPreFlight is called before PreBind, and determines whether PreBind is going to do something for this pod, or not.
// It just checks state.claims to determine whether there are any claims and hence the plugin has to handle them at PreBind.
func (pl *DynamicResources) PreBindPreFlight(ctx context.Context, cs fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if !pl.enabled {
		return fwk.NewStatus(fwk.Skip)
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return fwk.NewStatus(fwk.Skip)
	}
	return nil
}

// bindExtendedResources creates the requestMappings for the special extended resource claim
func (pl *DynamicResources) bindExtendedResources(claim *resourceapi.ResourceClaim, pod *v1.Pod) []*corev1apply.ContainerExtendedResourceRequestApplyConfiguration {
	var cer []*corev1apply.ContainerExtendedResourceRequestApplyConfiguration
	var deviceReqNames []string
	for _, r := range claim.Spec.Devices.Requests {
		deviceReqNames = append(deviceReqNames, r.Name)
	}
	containers := slices.Clone(pod.Spec.InitContainers)
	containers = append(containers, pod.Spec.Containers...)
	for i, c := range containers {
		creqs := c.Resources.Requests
		keys := make([]string, 0, len(creqs))
		for k := range creqs {
			keys = append(keys, k.String())
		}
		slice.SortStrings(keys)
		for rName := range creqs {
			ridx := 0
			for j := range keys {
				if keys[j] == rName.String() {
					ridx = j
					break
				}
			}
			for _, devReqName := range deviceReqNames {
				// During filter phase, device request name is set to be
				// container name index "-" extended resource name index
				if fmt.Sprintf("container-%d-request-%d", i, ridx) == devReqName {
					cer = append(cer,
						corev1apply.ContainerExtendedResourceRequest().
							WithContainerName(c.Name).
							WithExtendedResourceName(rName.String()).
							WithRequestName(devReqName))
				}
			}
		}
	}
	return cer
}

// bindClaim gets called by PreBind for claim which is not reserved for the pod yet.
// It might not even be allocated. bindClaim then ensures that the allocation
// and reservation are recorded. This finishes the work started in Reserve.
func (pl *DynamicResources) bindClaim(ctx context.Context, state *stateData, index int, pod *v1.Pod, nodeName string) (patchedClaim *resourceapi.ResourceClaim, finalErr error) {
	logger := klog.FromContext(ctx)
	claim := state.claims.get(index)
	allocation := state.informationsForClaim[index].allocation
	isExtendedResourceClaim := false
	if claim == state.claims.extendedResourceClaim() {
		// extended resource requests satisfied by device plugin
		if allocation == nil && claim.Spec.Devices.Requests == nil {
			return claim, nil
		}
		isExtendedResourceClaim = true
	}
	claimUIDs := []types.UID{claim.UID}
	defer func() {
		if allocation != nil {
			// The scheduler was handling allocation. Now that has
			// completed, either successfully or with a failure.
			if finalErr == nil {
				// This can fail, but only for reasons that are okay (concurrent delete or update).
				// Shouldn't happen in this case.
				if err := pl.draManager.ResourceClaims().AssumeClaimAfterAPICall(claim); err != nil {
					logger.V(5).Info("Claim not stored in assume cache", "err", finalErr)
				}
			}
			for _, claimUID := range claimUIDs {
				pl.draManager.ResourceClaims().RemoveClaimPendingAllocation(claimUID)
			}
		}
	}()

	// Create the special claim for extended resource backed by DRA?
	if isExtendedResourceClaim && claim.Name == specialClaimInMemName {
		// Replace claim template with instantiated claim for the node.
		if nodeAllocation, ok := state.nodeAllocations[nodeName]; ok && nodeAllocation.extendedResourceClaim != nil {
			claim = nodeAllocation.extendedResourceClaim.DeepCopy()
		} else {
			return nil, fmt.Errorf("extended resource claim not found for node %s", nodeName)
		}

		// Clear fields which must or can not be set during creation.
		claim.Status.Allocation = nil
		claim.Name = ""
		claim.UID = ""
		var err error
		claim, err = pl.clientset.ResourceV1beta1().ResourceClaims(claim.Namespace).Create(ctx, claim, metav1.CreateOptions{})
		if err != nil {
			return nil, fmt.Errorf("create claim for extended resources %s: %w", klog.KObj(claim), err)
		}
		logger.V(5).Info("created claim for extended resources", "pod", klog.KObj(pod), "node", klog.ObjectRef{Name: nodeName}, "resourceclaim", klog.Format(claim))

		// Track the actual extended ResourceClaim from now.
		// Relevant if we need to delete again in Unreserve.
		state.claims.set(state.claims.len()-1, claim)
	}

	logger.V(5).Info("preparing claim status update", "claim", klog.KObj(state.claims.get(index)), "allocation", klog.Format(allocation))

	// We may run into a ResourceVersion conflict because there may be some
	// benign concurrent changes. In that case we get the latest claim and
	// try again.
	refreshClaim := false
	claim = claim.DeepCopy()
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if refreshClaim {
			updatedClaim, err := pl.clientset.ResourceV1beta1().ResourceClaims(claim.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("get updated claim %s after conflict: %w", klog.KObj(claim), err)
			}
			logger.V(5).Info("retrying update after conflict", "claim", klog.KObj(claim))
			claim = updatedClaim
		} else {
			// All future retries must get a new claim first.
			refreshClaim = true
		}

		if claim.DeletionTimestamp != nil {
			return fmt.Errorf("claim %s got deleted in the meantime", klog.KObj(claim))
		}

		// Do we need to store an allocation result from Reserve?
		if allocation != nil {
			if claim.Status.Allocation != nil {
				return fmt.Errorf("claim %s got allocated elsewhere in the meantime", klog.KObj(claim))
			}

			// The finalizer needs to be added in a normal update.
			// If we were interrupted in the past, it might already be set and we simply continue.
			if !slices.Contains(claim.Finalizers, resourceapi.Finalizer) {
				claim.Finalizers = append(claim.Finalizers, resourceapi.Finalizer)
				updatedClaim, err := pl.clientset.ResourceV1beta1().ResourceClaims(claim.Namespace).Update(ctx, claim, metav1.UpdateOptions{})
				if err != nil {
					return fmt.Errorf("add finalizer to claim %s: %w", klog.KObj(claim), err)
				}
				claim = updatedClaim
			}
			claim.Status.Allocation = allocation
		}

		// We can simply try to add the pod here without checking
		// preconditions. The apiserver will tell us with a
		// non-conflict error if this isn't possible.
		claim.Status.ReservedFor = append(claim.Status.ReservedFor, resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: pod.Name, UID: pod.UID})
		updatedClaim, err := pl.clientset.ResourceV1beta1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
		if err != nil {
			if allocation != nil {
				return fmt.Errorf("add allocation and reservation to claim %s: %w", klog.KObj(claim), err)
			}
			return fmt.Errorf("add reservation to claim %s: %w", klog.KObj(claim), err)
		}
		claim = updatedClaim
		return nil
	})

	if retryErr != nil {
		return nil, retryErr
	}

	logger.V(5).Info("reserved", "pod", klog.KObj(pod), "node", klog.ObjectRef{Name: nodeName}, "resourceclaim", klog.Format(claim))

	// Patch the pod status with the new information about the generated
	// special resource claim.
	if isExtendedResourceClaim {
		cer := pl.bindExtendedResources(claim, pod)
		status := corev1apply.PodExtendedResourceClaimStatus().WithRequestMappings(cer...).WithResourceClaimName(claim.Name)
		podApply := corev1apply.Pod(pod.Name, pod.Namespace).WithStatus(corev1apply.PodStatus().WithExtendedResourceClaimStatus(status))
		retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if _, err := pl.clientset.CoreV1().Pods(pod.Namespace).ApplyStatus(ctx, podApply, metav1.ApplyOptions{FieldManager: fieldManager, Force: true}); err != nil {
				return fmt.Errorf("update pod %s/%s ExtendedResourceClaimStatus: %w", pod.Namespace, pod.Name, err)
			}
			return nil
		})
		if retryErr != nil {
			return nil, retryErr
		}
	}

	return claim, nil
}

// statusUnschedulable ensures that there is a log message associated with the
// line where the status originated.
func statusUnschedulable(logger klog.Logger, reason string, kv ...interface{}) *fwk.Status {
	if loggerV := logger.V(5); loggerV.Enabled() {
		helper, loggerV := loggerV.WithCallStackHelper()
		helper()
		kv = append(kv, "reason", reason)
		// nolint: logcheck // warns because it cannot check key/values
		loggerV.Info("pod unschedulable", kv...)
	}
	return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, reason)
}

// statusError ensures that there is a log message associated with the
// line where the error originated.
func statusError(logger klog.Logger, err error, kv ...interface{}) *fwk.Status {
	if loggerV := logger.V(5); loggerV.Enabled() {
		helper, loggerV := loggerV.WithCallStackHelper()
		helper()
		// nolint: logcheck // warns because it cannot check key/values
		loggerV.Error(err, "dynamic resource plugin failed", kv...)
	}
	return fwk.AsStatus(err)
}
