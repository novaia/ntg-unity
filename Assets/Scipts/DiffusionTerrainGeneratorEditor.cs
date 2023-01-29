using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(DiffusionTerrainGenerator))]
public class DiffusionTerrainGeneratorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        DiffusionTerrainGenerator myScript = (DiffusionTerrainGenerator)target;
        if(GUILayout.Button("Generate Terrain From Scratch"))
        {
            myScript.Setup();
            float[] heightmap = myScript.GenerateHeightmapFromScratch();
            myScript.SetTerrainHeights(heightmap);
        }
    }
}
