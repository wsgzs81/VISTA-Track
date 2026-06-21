#include "MVTrackAnnotationWriter.h"
#include "MVTrackRenderWriter.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"

namespace
{
FVector4f ClampBBoxToImage(const FVector4f& B, int32 ResX, int32 ResY)
{
    const float X0 = FMath::Clamp(B.X, 0.0f, (float)ResX);
    const float Y0 = FMath::Clamp(B.Y, 0.0f, (float)ResY);
    const float X1 = FMath::Clamp(B.X + B.Z, 0.0f, (float)ResX);
    const float Y1 = FMath::Clamp(B.Y + B.W, 0.0f, (float)ResY);
    return FVector4f(X0, Y0, FMath::Max(0.0f, X1 - X0), FMath::Max(0.0f, Y1 - Y0));
}

bool IsReasonableBBox(const FVector4f& B, int32 ResX, int32 ResY)
{
    if (!FMath::IsFinite(B.X) || !FMath::IsFinite(B.Y) ||
        !FMath::IsFinite(B.Z) || !FMath::IsFinite(B.W))
    {
        return false;
    }
    if (B.Z <= 0.0f || B.W <= 0.0f)
    {
        return false;
    }
    if (B.Z > ResX * 1.25f || B.W > ResY * 1.25f)
    {
        return false;
    }
    return (B.X + B.Z > 0.0f) && (B.X < ResX) &&
           (B.Y + B.W > 0.0f) && (B.Y < ResY);
}
}

FMVTrackFrameAnnotation UMVTrackAnnotationWriter::GenerateAnnotation(
    int32 FrameIndex, float Timestamp, const FString& CameraId,
    const FTransform& CameraTransform, float FOVDegrees,
    int32 ResX, int32 ResY,
    const FTransform& TargetTransform, const FBox& TargetWorldBounds,
    int32 TargetId, const FString& Category,
    bool bCameraOccluded)
{
    FMVTrackFrameAnnotation Ann;
    Ann.FrameIndex = FrameIndex;
    Ann.Timestamp = Timestamp;
    Ann.CameraId = CameraId;
    Ann.TargetId = TargetId;
    Ann.Category = Category;

    // Target pose
    Ann.TargetLocation = TargetTransform.GetLocation();
    Ann.TargetRotation = TargetTransform.GetRotation();
    Ann.TargetScale = TargetTransform.GetScale3D();

    // 3D bbox in world
    FVector Center, Extent;
    TargetWorldBounds.GetCenterAndExtents(Center, Extent);
    Ann.BBox3DCenter = Center;
    Ann.BBox3DExtent = Extent * 2.0f; // Full extent
    Ann.BBox3DRotation = TargetTransform.GetRotation();

    // Amodal bbox from 3D projection
    Ann.BBox2DAmodal = ProjectAmodalBBox(TargetWorldBounds, CameraTransform, FOVDegrees, ResX, ResY);
    Ann.AmodalProjectedArea = FMath::Max(0, (int32)(Ann.BBox2DAmodal.Z * Ann.BBox2DAmodal.W));

    // Visibility estimation
    if (bCameraOccluded)
    {
        Ann.VisibilityRatio = 0.55f;
        Ann.OcclusionState = EMVTrackVisibility::Partial;
        Ann.VisiblePixels = (int32)(Ann.AmodalProjectedArea * Ann.VisibilityRatio);
        Ann.BBox2DVisible = FVector4f(
            Ann.BBox2DAmodal.X,
            Ann.BBox2DAmodal.Y,
            Ann.BBox2DAmodal.Z * Ann.VisibilityRatio,
            Ann.BBox2DAmodal.W);
    }
    else
    {
        Ann.VisibilityRatio = 1.0f; // Full visibility when no occluder blocks
        Ann.OcclusionState = EMVTrackVisibility::Full;
        Ann.VisiblePixels = Ann.AmodalProjectedArea;
        Ann.BBox2DVisible = Ann.BBox2DAmodal;
    }

    // Check if target is within frame
    FVector4f ABB = Ann.BBox2DAmodal;
    bool bInFrame = (ABB.X + ABB.Z > 0) && (ABB.X < ResX) &&
                    (ABB.Y + ABB.W > 0) && (ABB.Y < ResY);
    Ann.bOutOfView = !bInFrame;
    Ann.bTruncated = bInFrame && (ABB.X < 0 || ABB.Y < 0 ||
                     ABB.X + ABB.Z > ResX || ABB.Y + ABB.W > ResY);

    Ann.bValid = bInFrame;
    if (!bInFrame) Ann.InvalidReason = TEXT("target_out_of_view");

    return Ann;
}

FMVTrackFrameAnnotation UMVTrackAnnotationWriter::GenerateAnnotationFromMask(
    int32 FrameIndex, float Timestamp, const FString& CameraId,
    const FTransform& CameraTransform, float FOVDegrees,
    int32 ResX, int32 ResY,
    const FTransform& TargetTransform, const FBox& TargetWorldBounds,
    int32 TargetId, const FString& Category,
    bool bCameraOccluded,
    const FMVTrackMaskStats& MaskStats)
{
    FMVTrackFrameAnnotation Ann = GenerateAnnotation(
        FrameIndex, Timestamp, CameraId, CameraTransform, FOVDegrees,
        ResX, ResY, TargetTransform, TargetWorldBounds,
        TargetId, Category, bCameraOccluded);

    if (MaskStats.bValid)
    {
        Ann.BBox2DVisible = ClampBBoxToImage(MaskStats.VisibleBBox, ResX, ResY);
        Ann.VisiblePixels = MaskStats.VisiblePixels;
        // The legacy 3D-corner amodal projection is unstable for close fixed
        // cameras and can be smaller than, or far outside, the mask bbox. For
        // SOT training, expose the mask-derived box as the stable target box.
        Ann.BBox2DAmodal = Ann.BBox2DVisible;
        Ann.BBox2DAmodal = ClampBBoxToImage(Ann.BBox2DAmodal, ResX, ResY);
        Ann.AmodalProjectedArea = FMath::Max(0, (int32)(Ann.BBox2DAmodal.Z * Ann.BBox2DAmodal.W));
        Ann.VisibilityRatio = Ann.AmodalProjectedArea > 0
            ? FMath::Clamp((float)MaskStats.VisiblePixels / (float)Ann.AmodalProjectedArea, 0.0f, 1.0f)
            : 0.0f;
        Ann.OcclusionState = ClassifyVisibility(Ann.VisibilityRatio);
        Ann.bValid = true;
        Ann.InvalidReason.Reset();
    }
    else
    {
        Ann.BBox2DVisible = FVector4f::Zero();
        Ann.BBox2DAmodal = FVector4f::Zero();
        Ann.AmodalProjectedArea = 0;
        Ann.VisiblePixels = 0;
        Ann.VisibilityRatio = 0.0f;
        Ann.OcclusionState = EMVTrackVisibility::Invisible;
        Ann.bValid = false;
        Ann.InvalidReason = TEXT("mask_empty");
    }

    FVector4f VBB = Ann.BBox2DVisible;
    bool bVisibleInFrame = MaskStats.bValid &&
        (VBB.X + VBB.Z > 0) && (VBB.X < ResX) &&
        (VBB.Y + VBB.W > 0) && (VBB.Y < ResY);
    Ann.bOutOfView = !bVisibleInFrame;
    Ann.bTruncated = bVisibleInFrame && (VBB.X <= 0 || VBB.Y <= 0 ||
                     VBB.X + VBB.Z >= ResX || VBB.Y + VBB.W >= ResY);
    return Ann;
}

FVector4f UMVTrackAnnotationWriter::ProjectAmodalBBox(
    const FBox& WorldBounds, const FTransform& CameraTransform,
    float FOVDegrees, int32 ResX, int32 ResY) const
{
    TArray<FVector> Corners = GetOBBcorners(WorldBounds, FTransform::Identity);

    float MinX = TNumericLimits<float>::Max();
    float MinY = TNumericLimits<float>::Max();
    float MaxX = TNumericLimits<float>::Lowest();
    float MaxY = TNumericLimits<float>::Lowest();

    for (const FVector& Corner : Corners)
    {
        FVector2D Screen = ProjectWorldToScreen(Corner, CameraTransform, FOVDegrees, ResX, ResY);
        MinX = FMath::Min(MinX, (float)Screen.X);
        MinY = FMath::Min(MinY, (float)Screen.Y);
        MaxX = FMath::Max(MaxX, (float)Screen.X);
        MaxY = FMath::Max(MaxY, (float)Screen.Y);
    }

    return FVector4f(MinX, MinY, MaxX - MinX, MaxY - MinY);
}

FVector2D UMVTrackAnnotationWriter::ProjectWorldToScreen(
    const FVector& WorldPoint, const FTransform& CameraTransform,
    float FOVDegrees, int32 ResX, int32 ResY)
{
    // Transform to camera space
    FVector LocalPoint = CameraTransform.InverseTransformPosition(WorldPoint);

    // UE camera looks along +X in local space
    if (LocalPoint.X <= 0.1f)
    {
        // Behind camera
        return FVector2D(-9999, -9999);
    }

    float FOVRad = FMath::DegreesToRadians(FOVDegrees);
    float HalfFOV = FOVRad * 0.5f;
    float TanHalfFOV = FMath::Tan(HalfFOV);

    // Project to normalized screen space
    float NX = (LocalPoint.Y / LocalPoint.X) / TanHalfFOV;
    float NY = -(LocalPoint.Z / LocalPoint.X) / TanHalfFOV; // Flip Y

    // Convert to pixel coordinates
    float PX = (NX * 0.5f + 0.5f) * ResX;
    float PY = (NY * 0.5f + 0.5f) * ResY;

    return FVector2D(PX, PY);
}

TArray<FVector> UMVTrackAnnotationWriter::GetOBBcorners(const FBox& Bounds, const FTransform& Transform)
{
    TArray<FVector> Corners;
    Corners.SetNum(8);
    FVector Min = Bounds.Min;
    FVector Max = Bounds.Max;

    for (int32 i = 0; i < 8; i++)
    {
        FVector Corner(
            (i & 1) ? Max.X : Min.X,
            (i & 2) ? Max.Y : Min.Y,
            (i & 4) ? Max.Z : Min.Z
        );
        Corners[i] = Transform.TransformPosition(Corner);
    }
    return Corners;
}

EMVTrackVisibility UMVTrackAnnotationWriter::ClassifyVisibility(float Ratio)
{
    if (Ratio >= 0.8f) return EMVTrackVisibility::Full;
    if (Ratio >= 0.4f) return EMVTrackVisibility::Partial;
    if (Ratio >= 0.1f) return EMVTrackVisibility::HeavyOcclusion;
    if (Ratio > 0.0f)  return EMVTrackVisibility::TinyVisible;
    return EMVTrackVisibility::Invisible;
}

float UMVTrackAnnotationWriter::EstimateVisibilityRatio(bool bOccluded, float OccluderCoverage)
{
    if (bOccluded) return FMath::Clamp(1.0f - OccluderCoverage, 0.0f, 1.0f);
    return 1.0f;
}

void UMVTrackAnnotationWriter::SaveAnnotationJson(const FMVTrackFrameAnnotation& Ann, const FString& FilePath)
{
    TSharedPtr<FJsonObject> Json = MakeShareable(new FJsonObject);
    Json->SetNumberField(TEXT("frame_index"), Ann.FrameIndex);
    Json->SetNumberField(TEXT("timestamp"), Ann.Timestamp);
    Json->SetStringField(TEXT("camera_id"), Ann.CameraId);
    Json->SetNumberField(TEXT("target_id"), Ann.TargetId);
    Json->SetStringField(TEXT("category"), Ann.Category);

    // BBox visible
    TArray<TSharedPtr<FJsonValue>> BBoxVis;
    BBoxVis.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DVisible.X)));
    BBoxVis.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DVisible.Y)));
    BBoxVis.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DVisible.Z)));
    BBoxVis.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DVisible.W)));
    Json->SetArrayField(TEXT("bbox_2d_visible_xywh"), BBoxVis);

    // BBox amodal
    TArray<TSharedPtr<FJsonValue>> BBoxAm;
    BBoxAm.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DAmodal.X)));
    BBoxAm.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DAmodal.Y)));
    BBoxAm.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DAmodal.Z)));
    BBoxAm.Add(MakeShareable(new FJsonValueNumber(Ann.BBox2DAmodal.W)));
    Json->SetArrayField(TEXT("bbox_2d_amodal_xywh"), BBoxAm);

    // 3D bbox
    TSharedPtr<FJsonObject> BBox3D = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> Center3D;
    Center3D.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DCenter.X)));
    Center3D.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DCenter.Y)));
    Center3D.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DCenter.Z)));
    BBox3D->SetArrayField(TEXT("center"), Center3D);

    TArray<TSharedPtr<FJsonValue>> Extent3D;
    Extent3D.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DExtent.X)));
    Extent3D.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DExtent.Y)));
    Extent3D.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DExtent.Z)));
    BBox3D->SetArrayField(TEXT("extent"), Extent3D);

    TArray<TSharedPtr<FJsonValue>> RotQ;
    RotQ.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DRotation.X)));
    RotQ.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DRotation.Y)));
    RotQ.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DRotation.Z)));
    RotQ.Add(MakeShareable(new FJsonValueNumber(Ann.BBox3DRotation.W)));
    BBox3D->SetArrayField(TEXT("rotation_quat"), RotQ);
    Json->SetObjectField(TEXT("bbox_3d_world"), BBox3D);

    // Pose
    TSharedPtr<FJsonObject> Pose = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> Loc;
    Loc.Add(MakeShareable(new FJsonValueNumber(Ann.TargetLocation.X)));
    Loc.Add(MakeShareable(new FJsonValueNumber(Ann.TargetLocation.Y)));
    Loc.Add(MakeShareable(new FJsonValueNumber(Ann.TargetLocation.Z)));
    Pose->SetArrayField(TEXT("location_cm"), Loc);
    Json->SetObjectField(TEXT("pose_world"), Pose);

    // Visibility
    TSharedPtr<FJsonObject> Vis = MakeShareable(new FJsonObject);
    Vis->SetNumberField(TEXT("visible_pixels"), Ann.VisiblePixels);
    Vis->SetNumberField(TEXT("visibility_ratio"), Ann.VisibilityRatio);

    FString OccStateStr;
    switch (Ann.OcclusionState)
    {
        case EMVTrackVisibility::Full:           OccStateStr = TEXT("full"); break;
        case EMVTrackVisibility::Partial:        OccStateStr = TEXT("partial"); break;
        case EMVTrackVisibility::HeavyOcclusion: OccStateStr = TEXT("heavy_occlusion"); break;
        case EMVTrackVisibility::TinyVisible:    OccStateStr = TEXT("tiny_visible"); break;
        case EMVTrackVisibility::Invisible:      OccStateStr = TEXT("invisible"); break;
    }
    Vis->SetStringField(TEXT("occlusion_state"), OccStateStr);
    Vis->SetBoolField(TEXT("truncated"), Ann.bTruncated);
    Vis->SetBoolField(TEXT("out_of_view"), Ann.bOutOfView);
    Json->SetObjectField(TEXT("visibility"), Vis);

    Json->SetBoolField(TEXT("valid"), Ann.bValid);
    if (!Ann.bValid) Json->SetStringField(TEXT("invalid_reason"), Ann.InvalidReason);

    // Write file
    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(FilePath), true);
    FFileHelper::SaveStringToFile(JsonStr, *FilePath);
}

void UMVTrackAnnotationWriter::SaveSequenceAnnotations(
    const TArray<FMVTrackFrameAnnotation>& Annotations,
    const FString& OutputDir)
{
    for (const auto& Ann : Annotations)
    {
        FString FilePath = FString::Printf(TEXT("%s/frames/%s/ann/%06d.json"),
            *OutputDir, *Ann.CameraId, Ann.FrameIndex);
        SaveAnnotationJson(Ann, FilePath);
    }
}
