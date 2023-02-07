using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(DiffusionTerrainGenerator))]
public class DiffusionTerrainGeneratorEditor : Editor
{
    DiffusionTerrainGenerator generator;

    SerializedProperty modelAsset;
    SerializedProperty modelOutputWidth;
    SerializedProperty modelOutputHeight;
    SerializedProperty terrain;
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
        modelAsset = serializedObject.FindProperty("modelAsset");
        modelOutputWidth = serializedObject.FindProperty("modelOutputWidth");
        modelOutputHeight = serializedObject.FindProperty("modelOutputHeight");
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

        EditorGUILayout.PropertyField(modelAsset);
        EditorGUILayout.PropertyField(modelOutputWidth);
        EditorGUILayout.PropertyField(modelOutputHeight);

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

        if(GUILayout.Button("Blend With Neighbors"))
        {
            generator.BlendWithNeighbors();
        }

        if(serializedObject.ApplyModifiedProperties())
        {
            generator.Setup();
        }
    }
}
