using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(GANTerrainGenerator))]
public class GANTerrainGeneratorEditor : Editor
{
    GANTerrainGenerator generator;

    SerializedProperty heightMultiplier;
    SerializedProperty modelAsset;
    SerializedProperty modelOutputWidth;
    SerializedProperty modelOutputHeight;

    // Latent boolean properties.
    SerializedProperty BigMountainTopLeft;
    SerializedProperty CentralValley;
    SerializedProperty Lowlands;
    SerializedProperty Highlands;
    SerializedProperty DiagonalRidge;
    SerializedProperty Highlands2;
    SerializedProperty CentralValley2;
    SerializedProperty BottomRightDecline;
    SerializedProperty BottomRightDecline2;
    SerializedProperty DivergingRidges;
    SerializedProperty Highlands3;
    SerializedProperty ValleyPass;
    SerializedProperty CentralValley3;
    SerializedProperty BottomLeftDecline;
    SerializedProperty randomNormal;

    public void OnEnable()
    {
        generator = (GANTerrainGenerator)target;
        generator.Setup();
        modelAsset = serializedObject.FindProperty("modelAsset");
        modelOutputWidth = serializedObject.FindProperty("modelOutputWidth");
        modelOutputHeight = serializedObject.FindProperty("modelOutputHeight");
        heightMultiplier = serializedObject.FindProperty("heightMultiplier");    

        // Latent boolean properties.
        BigMountainTopLeft = serializedObject.FindProperty("BigMountainTopLeft");
        CentralValley = serializedObject.FindProperty("CentralValley");
        Lowlands = serializedObject.FindProperty("Lowlands");
        Highlands = serializedObject.FindProperty("Highlands");
        DiagonalRidge = serializedObject.FindProperty("DiagonalRidge");
        Highlands2 = serializedObject.FindProperty("Highlands2");
        CentralValley2 = serializedObject.FindProperty("CentralValley2");
        BottomRightDecline = serializedObject.FindProperty("BottomRightDecline");
        BottomRightDecline2 = serializedObject.FindProperty("BottomRightDecline2");
        DivergingRidges = serializedObject.FindProperty("DivergingRidges");
        Highlands3 = serializedObject.FindProperty("Highlands3");
        ValleyPass = serializedObject.FindProperty("ValleyPass");
        CentralValley3 = serializedObject.FindProperty("CentralValley3");
        BottomLeftDecline = serializedObject.FindProperty("BottomLeftDecline");
        randomNormal = serializedObject.FindProperty("randomNormal");
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.PropertyField(modelAsset);
        EditorGUILayout.PropertyField(modelOutputWidth);
        EditorGUILayout.PropertyField(modelOutputHeight);
        EditorGUILayout.PropertyField(heightMultiplier);

        // Display latent boolean properties.
        EditorGUILayout.PropertyField(BigMountainTopLeft);
        EditorGUILayout.PropertyField(CentralValley);
        EditorGUILayout.PropertyField(Lowlands);
        EditorGUILayout.PropertyField(Highlands);
        EditorGUILayout.PropertyField(DiagonalRidge);
        EditorGUILayout.PropertyField(Highlands2);
        EditorGUILayout.PropertyField(CentralValley2);
        EditorGUILayout.PropertyField(BottomRightDecline);
        EditorGUILayout.PropertyField(BottomRightDecline2);
        EditorGUILayout.PropertyField(DivergingRidges);
        EditorGUILayout.PropertyField(Highlands3);
        EditorGUILayout.PropertyField(ValleyPass);
        EditorGUILayout.PropertyField(CentralValley3);
        EditorGUILayout.PropertyField(BottomLeftDecline);
        EditorGUILayout.PropertyField(randomNormal);

        if(GUILayout.Button("Generate Terrain"))
        {
            float[] heightmap = generator.GenerateHeightmapFromLatent();
            generator.SetTerrainHeights(heightmap);
        }

        if(serializedObject.ApplyModifiedProperties())
        {
            generator.Setup();
        }
    }
}
