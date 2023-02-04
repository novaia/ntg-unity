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
    SerializedProperty heightmapTexture;
    SerializedProperty heightMultiplier;
    SerializedProperty startingDiffusionIterationFromExisting;

    public void OnEnable()
    {
        generator = (DiffusionTerrainGenerator)target;
        generator.Setup();
        diffusionIterationsFromScratch = serializedObject.FindProperty("diffusionIterationsFromScratch");
        diffusionIterationsFromExisting = serializedObject.FindProperty("diffusionIterationsFromExisting");
        existingHeightmapWeight = serializedObject.FindProperty("existingHeightmapWeight");
        heightMultiplier = serializedObject.FindProperty("heightMultiplier");
        startingDiffusionIterationFromExisting = serializedObject.FindProperty("startingDiffusionIterationFromExisting");
    }

    public override void OnInspectorGUI()
    {
        //DrawDefaultInspector();
        serializedObject.Update();

        if(GUILayout.Button("Clear Terrain"))
        {
            generator.ClearTerrain();
        }

        EditorGUILayout.PropertyField(heightMultiplier);
        EditorGUILayout.PropertyField(diffusionIterationsFromScratch);

        if(GUILayout.Button("Generate Terrain From Scratch"))
        {
            float[] heightmap = generator.GenerateHeightmapFromScratch();
            generator.SetTerrainHeights(heightmap);
        }

        EditorGUILayout.PropertyField(diffusionIterationsFromExisting);
        EditorGUILayout.PropertyField(startingDiffusionIterationFromExisting);
        EditorGUILayout.Slider(existingHeightmapWeight, 1, 0);

        GUILayout.Box(generator.GetTerrainHeightmapAsTexture());

        if(GUILayout.Button("Generate Terrain From Existing"))
        {
            float[] heightmap = generator.GenerateHeightmapFromExisting();
            generator.SetTerrainHeights(heightmap);
        }

        serializedObject.ApplyModifiedProperties();
    }
}
