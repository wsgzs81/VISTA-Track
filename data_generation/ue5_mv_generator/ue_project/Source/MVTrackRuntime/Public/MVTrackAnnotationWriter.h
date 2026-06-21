// MVTrackAnnotationWriter.h — Per-frame annotation generation and file output
#pragma once

#include "CoreMinimal.h"
#include "MVTrackTypes.h"
#include "MVTrackAnnotationWriter.generated.h"

class USceneCaptureComponent2D;
struct FMVTrackMaskStats;

/**
 * Generates per-frame per-camera annotations including:
 * - Visible bbox (from instance mask)
 * - Amodal bbox (from 3D bbox projection)
 * - 3D bbox in world and camera coordinates
 * - Visibility ratio and occlusion state
 * - Camera calibration
 */
UCLASS()
class MVTRACKRUNTIME_API UMVTrackAnnotationWriter : public UActorComponent
{
    GENERATED_BODY()

public:
    /** Generate annotation for one camera at one frame */
    FMVTrackFrameAnnotation GenerateAnnotation(
        int32 FrameIndex,
        float Timestamp,
        const FString& CameraId,
        const FTransform& CameraTransform,
        float FOVDegrees,
        int32 ResX, int32 ResY,
        const FTransform& TargetTransform,
        const FBox& TargetWorldBounds,
        int32 TargetId,
        const FString& Category,
        bool bCameraOccluded);

    FMVTrackFrameAnnotation GenerateAnnotationFromMask(
        int32 FrameIndex,
        float Timestamp,
        const FString& CameraId,
        const FTransform& CameraTransform,
        float FOVDegrees,
        int32 ResX, int32 ResY,
        const FTransform& TargetTransform,
        const FBox& TargetWorldBounds,
        int32 TargetId,
        const FString& Category,
        bool bCameraOccluded,
        const FMVTrackMaskStats& MaskStats);

    /** Compute 2D amodal bbox by projecting 3D bbox corners to screen */
    FVector4f ProjectAmodalBBox(
        const FBox& WorldBounds,
        const FTransform& CameraTransform,
        float FOVDegrees,
        int32 ResX, int32 ResY) const;

    /** Determine visibility state from ratio */
    static EMVTrackVisibility ClassifyVisibility(float Ratio);

    /** Save a single annotation as JSON */
    static void SaveAnnotationJson(const FMVTrackFrameAnnotation& Ann, const FString& FilePath);

    /** Save all frame annotations for a sequence */
    static void SaveSequenceAnnotations(const TArray<FMVTrackFrameAnnotation>& Annotations,
                                         const FString& OutputDir);

    /** Compute visibility ratio (placeholder; real impl needs mask pixel count) */
    static float EstimateVisibilityRatio(bool bOccluded, float OccluderCoverage);

    /** Project a world point to screen coordinates */
    static FVector2D ProjectWorldToScreen(const FVector& WorldPoint,
                                           const FTransform& CameraTransform,
                                           float FOVDegrees,
                                           int32 ResX, int32 ResY);

    /** Get 8 corners of an oriented bounding box */
    static TArray<FVector> GetOBBcorners(const FBox& Bounds, const FTransform& Transform);
};
