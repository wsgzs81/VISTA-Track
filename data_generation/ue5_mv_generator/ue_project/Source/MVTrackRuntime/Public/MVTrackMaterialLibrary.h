#pragma once

#include "CoreMinimal.h"

class UObject;
class UStaticMeshComponent;

namespace MVTrackMaterials
{
enum class EMaterialRole : uint8
{
    Target,
    Furniture,
    Occluder,
    Floor,
    Wall,
    Prop
};

void ApplySemanticMaterial(
    UStaticMeshComponent* MeshComp,
    UObject* Outer,
    const FString& Category,
    int32 StyleSeed,
    EMaterialRole Role);

}
