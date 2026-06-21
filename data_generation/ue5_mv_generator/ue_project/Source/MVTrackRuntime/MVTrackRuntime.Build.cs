using UnrealBuildTool;

public class MVTrackRuntime : ModuleRules
{
    public MVTrackRuntime(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "RenderCore",
            "RHI",
            "ImageWriteQueue",
            "Json",
            "JsonUtilities",
            "PhysicsCore",
            "Chaos",
            "ChaosCore",
            "GeometryCore"
        });

        PrivateDependencyModuleNames.AddRange(new string[] {
            "Slate",
            "SlateCore",
            "LevelSequence",
            "MovieScene",
            "MovieSceneCapture"
        });
    }
}
