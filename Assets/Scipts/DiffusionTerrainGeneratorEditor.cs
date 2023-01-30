using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(DiffusionTerrainGenerator))]
public class DiffusionTerrainGeneratorEditor : Editor
{
    DiffusionTerrainGenerator generator;

    SerializedProperty diffusionIterationsFromScratch;
    SerializedProperty diffusionIterationsFromExisting;
    SerializedProperty existingHeightmapWeight;
    SerializedProperty noiseWeight;
    SerializedProperty heightmapTexture;

    public void OnEnable()
    {
        generator = (DiffusionTerrainGenerator)target;
        generator.Setup();
        diffusionIterationsFromScratch = serializedObject.FindProperty("diffusionIterationsFromScratch");
        diffusionIterationsFromExisting = serializedObject.FindProperty("diffusionIterationsFromExisting");
        existingHeightmapWeight = serializedObject.FindProperty("existingHeightmapWeight");
        noiseWeight = serializedObject.FindProperty("noiseWeight");
    }

    public override void OnInspectorGUI()
    {
        //DrawDefaultInspector();

        if(GUILayout.Button("Clear Terrain"))
        {
            generator.ClearTerrain();
        }


        serializedObject.Update();
        EditorGUILayout.PropertyField(diffusionIterationsFromScratch);

        if(GUILayout.Button("Generate Terrain From Scratch"))
        {
            float[] heightmap = generator.GenerateHeightmapFromScratch();
            generator.SetTerrainHeights(heightmap);
        }

        EditorGUILayout.PropertyField(diffusionIterationsFromExisting);
        //EditorGUILayout.PropertyField(existingHeightmapWeight);
        //EditorGUILayout.PropertyField(noiseWeight);
        EditorGUILayout.Slider(existingHeightmapWeight, 1, 0);
        EditorGUILayout.Slider(noiseWeight, 1, 0);

        //GUILayout.Box(generator.terrain.terrainData.heightmapTexture);
        GUILayout.Box(generator.GetTerrainHeightmapAsTexture());

        if(GUILayout.Button("Generate Terrain From Existing"))
        {
            float[] heightmap = generator.GenerateHeightmapFromExisting();
            generator.SetTerrainHeights(heightmap);
        }

        serializedObject.ApplyModifiedProperties();
    }
}
