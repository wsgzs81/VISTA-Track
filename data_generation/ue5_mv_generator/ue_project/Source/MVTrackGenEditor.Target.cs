using UnrealBuildTool;
using System.Collections.Generic;

public class MVTrackGenEditorTarget : TargetRules
{
    public MVTrackGenEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V4;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_4;
        ExtraModuleNames.Add("MVTrackGen");
    }
}
