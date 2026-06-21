// MVTrackTypes.h — Shared data types for MVTrack dataset generation
#pragma once

#include "CoreMinimal.h"
#include "MVTrackTypes.generated.h"

/** Visibility state of target from a camera */
UENUM(BlueprintType)
enum class EMVTrackVisibility : uint8
{
    Full            UMETA(DisplayName = "Full"),
    Partial         UMETA(DisplayName = "Partial"),
    HeavyOcclusion  UMETA(DisplayName = "Heavy Occlusion"),
    TinyVisible     UMETA(DisplayName = "Tiny Visible"),
    Invisible       UMETA(DisplayName = "Invisible")
};

/** Failure reason for sequence generation */
UENUM(BlueprintType)
enum class EMVTrackFailure : uint8
{
    None                    UMETA(DisplayName = "None"),
    AssetImportFailed       UMETA(DisplayName = "Asset Import Failed"),
    CollisionGenFailed      UMETA(DisplayName = "Collision Gen Failed"),
    SettlementFailed        UMETA(DisplayName = "Settlement Failed"),
    TargetPenetration       UMETA(DisplayName = "Target Penetration"),
    TargetOutOfBounds       UMETA(DisplayName = "Target Out Of Bounds"),
    NoVisibleCamera         UMETA(DisplayName = "No Visible Camera"),
    RenderTimeout           UMETA(DisplayName = "Render Timeout"),
    MissingFrame            UMETA(DisplayName = "Missing Frame"),
    MaskEmpty               UMETA(DisplayName = "Mask Empty"),
    BBoxMaskMismatch        UMETA(DisplayName = "BBox Mask Mismatch"),
    DepthInvalid            UMETA(DisplayName = "Depth Invalid"),
    UECrash                 UMETA(DisplayName = "UE Crash")
};

/** Per-camera per-frame annotation data */
USTRUCT(BlueprintType)
struct FMVTrackFrameAnnotation
{
    GENERATED_BODY()

    UPROPERTY() int32 FrameIndex = 0;
    UPROPERTY() float Timestamp = 0.0f;
    UPROPERTY() FString CameraId;
    UPROPERTY() int32 TargetId = 1;
    UPROPERTY() FString Category;

    // 2D visible bbox (xywh in pixels, from mask bounding rect)
    UPROPERTY() FVector4f BBox2DVisible = FVector4f::Zero();
    // 2D amodal bbox (xywh in pixels, from 3D bbox projection)
    UPROPERTY() FVector4f BBox2DAmodal = FVector4f::Zero();

    // 3D bbox in world coordinates
    UPROPERTY() FVector BBox3DCenter = FVector::ZeroVector;
    UPROPERTY() FVector BBox3DExtent = FVector::ZeroVector;
    UPROPERTY() FQuat BBox3DRotation = FQuat::Identity;

    // Target pose in world
    UPROPERTY() FVector TargetLocation = FVector::ZeroVector;
    UPROPERTY() FQuat TargetRotation = FQuat::Identity;
    UPROPERTY() FVector TargetScale = FVector::OneVector;

    // Visibility
    UPROPERTY() int32 VisiblePixels = 0;
    UPROPERTY() int32 AmodalProjectedArea = 0;
    UPROPERTY() float VisibilityRatio = 0.0f;
    UPROPERTY() EMVTrackVisibility OcclusionState = EMVTrackVisibility::Full;
    UPROPERTY() bool bTruncated = false;
    UPROPERTY() bool bOutOfView = false;

    // Validity
    UPROPERTY() bool bValid = true;
    UPROPERTY() FString InvalidReason;
};

/** Camera intrinsics + extrinsics for one camera at one frame */
USTRUCT(BlueprintType)
struct FMVTrackCameraCalibration
{
    GENERATED_BODY()

    UPROPERTY() FString CameraId;
    UPROPERTY() int32 ResolutionX = 1280;
    UPROPERTY() int32 ResolutionY = 720;
    UPROPERTY() float SensorWidthMM = 36.0f;
    UPROPERTY() float SensorHeightMM = 20.25f;
    UPROPERTY() float FocalLengthMM = 35.0f;
    UPROPERTY() FVector2D PrincipalPoint = FVector2D(640.0, 360.0);

    // K matrix (3x3 stored as 9 floats, row-major)
    UPROPERTY() TArray<float> KMatrix;

    // Camera-to-world transform
    UPROPERTY() FTransform CameraToWorld = FTransform::Identity;
    // World-to-camera transform
    UPROPERTY() FTransform WorldToCamera = FTransform::Identity;
};

/** Job manifest — the complete specification for one sequence */
USTRUCT(BlueprintType)
struct FMVTrackJobManifest
{
    GENERATED_BODY()

    UPROPERTY() FString JobId;
    UPROPERTY() FString SequenceId;
    UPROPERTY() int32 SequenceIndex = 0;
    UPROPERTY() int32 Seed = 0;
    UPROPERTY() FString TargetCategory;
    UPROPERTY() FString TargetMeshPath;
    UPROPERTY() float TargetGroundZCm = 0.0f;
    UPROPERTY() float TargetScaleM = 0.3f;
    UPROPERTY() int32 NumCameras = 4;
    UPROPERTY() int32 NumFrames = 300;
    UPROPERTY() int32 FPS = 30;
    UPROPERTY() int32 ResolutionX = 1280;
    UPROPERTY() int32 ResolutionY = 720;
    UPROPERTY() int32 NumOccluders = 4;
    UPROPERTY() FString OutputDir;
    UPROPERTY() FString MotionType;
};
