#include "MVTrackMaterialLibrary.h"

#include "Components/StaticMeshComponent.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Math/RandomStream.h"
#include "Misc/Crc.h"

namespace
{
struct FMaterialLook
{
    TArray<FLinearColor> Palette;
    float Variation = 0.08f;
    float Roughness = 0.75f;
    float Metallic = 0.0f;
};

FLinearColor C(float R, float G, float B)
{
    return FLinearColor(R, G, B, 1.0f);
}

void Add(TArray<FLinearColor>& Out, std::initializer_list<FLinearColor> Colors)
{
    for (const FLinearColor& Color : Colors)
    {
        Out.Add(Color);
    }
}

FMaterialLook BuildLook(const FString& Category, MVTrackMaterials::EMaterialRole Role)
{
    FString Key = Category.ToLower();
    FMaterialLook Look;

    if (Role == MVTrackMaterials::EMaterialRole::Floor)
    {
        Add(Look.Palette, {
            C(0.48f, 0.34f, 0.20f),
            C(0.36f, 0.24f, 0.14f),
            C(0.62f, 0.47f, 0.30f),
            C(0.28f, 0.22f, 0.17f),
        });
        Look.Variation = 0.05f;
        Look.Roughness = 0.82f;
        return Look;
    }

    if (Role == MVTrackMaterials::EMaterialRole::Wall)
    {
        Add(Look.Palette, {
            C(0.82f, 0.79f, 0.72f),
            C(0.76f, 0.79f, 0.78f),
            C(0.70f, 0.73f, 0.68f),
            C(0.86f, 0.82f, 0.76f),
        });
        Look.Variation = 0.035f;
        Look.Roughness = 0.9f;
        return Look;
    }

    if (Role == MVTrackMaterials::EMaterialRole::Occluder)
    {
        Add(Look.Palette, {
            C(0.55f, 0.43f, 0.31f),
            C(0.43f, 0.37f, 0.31f),
            C(0.64f, 0.59f, 0.51f),
            C(0.32f, 0.30f, 0.28f),
            C(0.50f, 0.32f, 0.22f),
        });
        Look.Variation = 0.07f;
        Look.Roughness = 0.86f;
        return Look;
    }

    if (Key.Contains(TEXT("television")) || Key.Contains(TEXT("tv")) || Key.Contains(TEXT("monitor")))
    {
        Add(Look.Palette, {
            C(0.015f, 0.017f, 0.020f),
            C(0.05f, 0.055f, 0.065f),
            C(0.12f, 0.13f, 0.14f),
            C(0.02f, 0.035f, 0.055f),
        });
        Look.Variation = 0.025f;
        Look.Roughness = 0.42f;
        Look.Metallic = 0.15f;
        return Look;
    }

    if (Key.Contains(TEXT("wicker")) || Key.Contains(TEXT("basket")))
    {
        Add(Look.Palette, {
            C(0.74f, 0.58f, 0.34f),
            C(0.58f, 0.39f, 0.20f),
            C(0.86f, 0.70f, 0.43f),
            C(0.42f, 0.27f, 0.13f),
            C(0.69f, 0.52f, 0.30f),
        });
        Look.Variation = 0.09f;
        Look.Roughness = 0.92f;
        return Look;
    }

    if (Key.Contains(TEXT("wood")) || Key.Contains(TEXT("table")) || Key.Contains(TEXT("stool")) ||
        Key.Contains(TEXT("cabinet")) || Key.Contains(TEXT("commode")) || Key.Contains(TEXT("bench")))
    {
        Add(Look.Palette, {
            C(0.50f, 0.29f, 0.13f),
            C(0.68f, 0.44f, 0.22f),
            C(0.36f, 0.20f, 0.10f),
            C(0.78f, 0.60f, 0.38f),
            C(0.32f, 0.25f, 0.19f),
        });
        if (Key.Contains(TEXT("painted")))
        {
            Add(Look.Palette, {
                C(0.58f, 0.66f, 0.62f),
                C(0.72f, 0.68f, 0.56f),
                C(0.45f, 0.52f, 0.56f),
                C(0.62f, 0.38f, 0.30f),
            });
        }
        Look.Variation = 0.075f;
        Look.Roughness = 0.78f;
        return Look;
    }

    if (Key.Contains(TEXT("sofa")) || Key.Contains(TEXT("chair")) || Key.Contains(TEXT("seat")) ||
        Key.Contains(TEXT("arm")))
    {
        Add(Look.Palette, {
            C(0.21f, 0.22f, 0.23f),
            C(0.37f, 0.34f, 0.30f),
            C(0.24f, 0.32f, 0.28f),
            C(0.43f, 0.30f, 0.24f),
            C(0.25f, 0.32f, 0.42f),
        });
        Look.Variation = 0.06f;
        Look.Roughness = 0.95f;
        return Look;
    }

    if (Key.Contains(TEXT("pomegranate")))
    {
        Add(Look.Palette, {
            C(0.55f, 0.05f, 0.05f),
            C(0.35f, 0.02f, 0.025f),
            C(0.70f, 0.12f, 0.08f),
            C(0.22f, 0.08f, 0.04f),
        });
        Look.Variation = 0.08f;
        Look.Roughness = 0.68f;
        return Look;
    }

    if (Key.Contains(TEXT("ginger")))
    {
        Add(Look.Palette, {
            C(0.76f, 0.58f, 0.34f),
            C(0.58f, 0.43f, 0.25f),
            C(0.86f, 0.70f, 0.45f),
            C(0.44f, 0.32f, 0.18f),
        });
        Look.Variation = 0.10f;
        Look.Roughness = 0.9f;
        return Look;
    }

    if (Key.Contains(TEXT("hamburger")) || Key.Contains(TEXT("bun")) || Key.Contains(TEXT("bread")))
    {
        Add(Look.Palette, {
            C(0.82f, 0.55f, 0.22f),
            C(0.95f, 0.73f, 0.38f),
            C(0.52f, 0.30f, 0.12f),
            C(0.74f, 0.45f, 0.18f),
        });
        Look.Variation = 0.08f;
        Look.Roughness = 0.86f;
        return Look;
    }

    if (Key.Contains(TEXT("metal")) || Key.Contains(TEXT("connector")) || Key.Contains(TEXT("bolt")))
    {
        Add(Look.Palette, {
            C(0.42f, 0.43f, 0.42f),
            C(0.22f, 0.23f, 0.24f),
            C(0.62f, 0.60f, 0.55f),
            C(0.12f, 0.13f, 0.14f),
        });
        Look.Variation = 0.04f;
        Look.Roughness = 0.45f;
        Look.Metallic = 0.35f;
        return Look;
    }

    Add(Look.Palette, {
        C(0.50f, 0.46f, 0.39f),
        C(0.34f, 0.38f, 0.35f),
        C(0.42f, 0.32f, 0.27f),
        C(0.28f, 0.33f, 0.42f),
        C(0.62f, 0.55f, 0.46f),
    });
    Look.Variation = 0.065f;
    Look.Roughness = 0.82f;
    return Look;
}

FLinearColor JitterColor(const FLinearColor& In, FRandomStream& RNG, float Amount)
{
    auto JitterChannel = [&](float V)
    {
        float Scale = RNG.FRandRange(1.0f - Amount, 1.0f + Amount);
        float Offset = RNG.FRandRange(-Amount * 0.035f, Amount * 0.035f);
        return FMath::Clamp(V * Scale + Offset, 0.005f, 0.98f);
    };

    return FLinearColor(
        JitterChannel(In.R),
        JitterChannel(In.G),
        JitterChannel(In.B),
        1.0f);
}
}

void MVTrackMaterials::ApplySemanticMaterial(
    UStaticMeshComponent* MeshComp,
    UObject* Outer,
    const FString& Category,
    int32 StyleSeed,
    EMaterialRole Role)
{
    if (!MeshComp)
    {
        return;
    }

    UMaterial* Base = LoadObject<UMaterial>(
        UMaterial::StaticClass(),
        TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
    if (!Base)
    {
        return;
    }

    const FMaterialLook Look = BuildLook(Category, Role);
    if (Look.Palette.Num() == 0)
    {
        return;
    }

    const uint32 Hash = FCrc::StrCrc32(*Category);
    FRandomStream LocalRNG(StyleSeed ^ static_cast<int32>(Hash));
    const int32 NumSlots = FMath::Max(1, MeshComp->GetNumMaterials());
    const int32 Start = LocalRNG.RandRange(0, Look.Palette.Num() - 1);

    for (int32 Slot = 0; Slot < NumSlots; ++Slot)
    {
        UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(
            Base,
            Outer ? Outer : MeshComp);
        if (!MID)
        {
            continue;
        }

        const FLinearColor BaseColor = Look.Palette[(Start + Slot) % Look.Palette.Num()];
        const FLinearColor Color = JitterColor(BaseColor, LocalRNG, Look.Variation);
        MID->SetVectorParameterValue(TEXT("Color"), Color);
        MID->SetVectorParameterValue(TEXT("BaseColor"), Color);
        MID->SetScalarParameterValue(TEXT("Roughness"), Look.Roughness);
        MID->SetScalarParameterValue(TEXT("Metallic"), Look.Metallic);
        MeshComp->SetMaterial(Slot, MID);
    }
}
