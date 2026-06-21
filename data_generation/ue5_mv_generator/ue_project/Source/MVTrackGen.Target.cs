using UnrealBuildTool;
using System.Collections.Generic;

public class MVTrackGenTarget : TargetRules
{
    public MVTrackGenTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V4;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_4;
        ExtraModuleNames.Add("MVTrackGen");
    }
}
