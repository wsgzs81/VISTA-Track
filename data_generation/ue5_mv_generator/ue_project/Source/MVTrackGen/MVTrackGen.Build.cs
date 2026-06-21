using UnrealBuildTool;

public class MVTrackGen : ModuleRules
{
    public MVTrackGen(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "MVTrackRuntime"
        });

        PrivateDependencyModuleNames.AddRange(new string[] {});
    }
}
