// MVTrackTargetController.h — Target spawning, physics settlement, motion, occluders
#pragma once

#include "CoreMinimal.h"
#include "MVTrackTypes.h"
#include "MVTrackTargetController.generated.h"

class UStaticMeshComponent;
class UPrimitiveComponent;

/**
 * Controls the target object lifecycle:
 * - Spawn with physics
 * - Settlement (gravity drop + wait for rest)
 * - Penetration check
 * - Motion (impulse-based rolling/sliding)
 * - Occluder placement (between camera and target to guarantee partial occlusion)
 */
UCLASS()
class MVTRACKRUNTIME_API UMVTrackTargetController : public UActorComponent
{
    GENERATED_BODY()

public:
    /** Spawn the target mesh with physics enabled */
    bool SpawnTarget(const FMVTrackJobManifest& Manifest,
                     const FVector& SpawnLocation,
                     FRandomStream& RNG);

    /** Spawn occluder objects to guarantee at least 1 camera is partially occluded */
    bool SpawnOccluders(const FMVTrackJobManifest& Manifest,
                        const FVector& TargetLocation,
                        const TArray<FVector>& CameraLocations,
                        FRandomStream& RNG);

    /** Run physics settlement: wait for object to come to rest */
    bool RunSettlement(float SettleTimeSec, float LinearVelThreshold, float AngularVelThreshold);

    /** Check for penetration with scene geometry */
    bool CheckPenetration() const;

    /** Apply motion impulse for the current frame */
    void ApplyMotion(int32 FrameIndex, FRandomStream& RNG);

    /** Get current target world transform */
    FTransform GetTargetTransform() const;

    /** Get target bounding box in world space */
    FBox GetTargetWorldBounds() const;

    /** Get target mesh component */
    UPROPERTY() UStaticMeshComponent* TargetMesh = nullptr;

    /** The spawned target actor */
    UPROPERTY() AActor* TargetActor = nullptr;

    /** All spawned occluder actors */
    UPROPERTY() TArray<AActor*> OccluderActors;

    /** Check if settlement succeeded */
    bool bSettlementSucceeded = false;

    /** Destroy all spawned actors */
    void Cleanup();

private:
    /** Create a primitive mesh (cube/sphere/cylinder/cone/torus) */
    AActor* SpawnPrimitiveMesh(const FString& MeshType,
                               const FVector& Location,
                               const FVector& Scale,
                               bool bPhysicsEnabled,
                               bool bIsDynamic,
                               FRandomStream& RNG);

    /** Place occluder between a camera and the target */
    FVector ComputeOccluderPosition(const FVector& CameraPos,
                                     const FVector& TargetPos,
                                     float DistanceFraction,
                                     float LateralOffset,
                                     FRandomStream& RNG);

    float MotionImpulseStrength = 200.0f;
    int32 LastImpulseFrame = -100;
    int32 ImpulseIntervalFrames = 60;
};
