using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEditor;
using UnityEditor.TerrainTools;
using UnityEngine.TerrainTools;

public class DiffusionGenerator : TerrainPaintTool<DiffusionGenerator>
{
    // General.
    private TensorMathHelper tensorMathHelper = new TensorMathHelper();
    private TerrainHelper terrainHelper = new TerrainHelper();
    private int modelOutputWidth = 256;
    private int modelOutputHeight = 256;
    private int upSampledWidth = 0;
    private int upSampledHeight = 0;
    private float heightMultiplier = 0.3f;
    private int channels = 1;
    private NNModel modelAsset;
    private Model runtimeModel;

    // Blending.
    private float radius1 = 128.0f;
    private float radius2 = 256.0f;
    private float bValue = 2.5f;
    private bool keepNeighborHeights;
    private NeighborBlender neighborBlender = new NeighborBlender();
    
    // Diffusion.
    private const float maxSignalRate = 0.9f;
    private const float minSignalRate = 0.02f;
    private Diffuser diffuser = new Diffuser();

    // Upsampling.
    private enum UpSampleMode {Bicubic};
    private UpSampleMode upSampleMode = UpSampleMode.Bicubic;
    // Left: upsample resolution, right: upsample factor.
    private enum UpSampleResolution 
    {
        _256 = 1, 
        _512 = 2, 
        _1024 = 4, 
        _2048 = 8, 
        _4096 = 16
    };
    private UpSampleResolution upSampleResolution = UpSampleResolution._512;
    private BicbubicUpSampler bicubicUpSampler = new BicbubicUpSampler();

    // Brushes.
    private bool brushesEnabled = false;
    private Texture2D brushTexture1;
    private Texture2D brushTexture2;
    private float m_BrushOpacity = 0.1f;
    private float m_BrushSize = 25f;
    private float m_BrushRotation = 0f;
    private Texture2D brushMask;
    private Texture2D brushHeightmap;
    private Texture2D brushHeightmapMasked;

    public override string GetName()
    {
        return "NTG/" + "Diffusion Generator";
    }

    public override string GetDescription()
    {
        return "Diffusion based neural terrain generator.";
    }

    private void GenerateMaskedBrushHeightmap()
    {
        if(brushHeightmap == null || brushMask == null) { return; }

        brushHeightmapMasked = new Texture2D(brushHeightmap.width, brushHeightmap.height);
        Color[] brushHeightmapColors = brushHeightmap.GetPixels();
        Color[] brushMaskColors = brushMask.GetPixels();
        Color[] brushHeightmapMaskedColors = new Color[brushHeightmapColors.Length];
        for(int i = 0; i < brushHeightmapColors.Length; i++)
        {
            brushHeightmapMaskedColors[i] = brushHeightmapColors[i] * brushMaskColors[i].r;
        }
        brushHeightmapMasked.SetPixels(brushHeightmapMaskedColors);
        brushHeightmapMasked.Apply();
    }

    public override void OnInspectorGUI(Terrain terrain, IOnInspectorGUI editContext)
    {
        BrushGUI();

        modelAsset = (NNModel)EditorGUILayout.ObjectField("Model Asset", modelAsset, typeof(NNModel), false);
        modelOutputWidth = EditorGUILayout.IntField("Model Output Width", modelOutputWidth);
        modelOutputHeight = EditorGUILayout.IntField("Model Output Height", modelOutputHeight);
        heightMultiplier = EditorGUILayout.FloatField("Height Multiplier", heightMultiplier);
        upSampleMode = (UpSampleMode)EditorGUILayout.EnumPopup("UpSample Mode", upSampleMode);
        upSampleResolution = (UpSampleResolution)EditorGUILayout.EnumPopup("UpSample Resolution", upSampleResolution);
        CalculateUpSampledDimensions();
        CalculateBlendingRadii();

        if(GUILayout.Button("Generate Terrain From Scratch"))
        {
            float[] heightmap = GenerateHeightmap();
            terrainHelper.SetTerrainHeights(terrain, heightmap, upSampledWidth, upSampledHeight, heightMultiplier);
        }

        bValue = EditorGUILayout.FloatField("B Value", bValue);
        keepNeighborHeights = EditorGUILayout.Toggle("Keep Neighbor Heights", keepNeighborHeights);

        if(GUILayout.Button("Blend With Neighbors"))
        {
            neighborBlender.BlendAllNeighbors(
                terrain, 
                upSampledWidth, 
                upSampledHeight, 
                radius1, 
                radius2, 
                bValue, 
                keepNeighborHeights
            );
        }
    }

    private void BrushGUI()
    {
        if(!brushesEnabled)
        {
            if(GUILayout.Button("Enable Brushes"))
            {
                brushesEnabled = true;
            }
        }
        else
        {
            if(GUILayout.Button("Disable Brushes"))
            {
                brushesEnabled = false;
            }

            Texture2D tempBrushMask = (Texture2D)EditorGUILayout.ObjectField("Brush Mask", brushMask, typeof(Texture2D), false);
            if(tempBrushMask)
            {
                brushMask = tempBrushMask;
                GenerateMaskedBrushHeightmap();
            }

            m_BrushOpacity = EditorGUILayout.Slider("Opacity", m_BrushOpacity, 0, 1);
            m_BrushSize = EditorGUILayout.Slider("Size", m_BrushSize, .001f, 2000f);
            m_BrushRotation = EditorGUILayout.Slider("Rotation", m_BrushRotation, 0, 360);

            if(GUILayout.Button("Generate Brush Heighmap"))
            {
                float[] brushHeightmapArray = GenerateHeightmap();
                Color[] colorBrushHeightmap = new Color[brushHeightmapArray.Length];
                for(int i = 0; i < brushHeightmapArray.Length; i++)
                {
                    colorBrushHeightmap[i] = new Color(brushHeightmapArray[i], brushHeightmapArray[i], brushHeightmapArray[i]);
                }
                brushHeightmap = new Texture2D(upSampledWidth, upSampledHeight);
                brushHeightmap.SetPixels(0, 0, upSampledWidth, upSampledHeight, colorBrushHeightmap);
                brushHeightmap.Apply();
            }
            if(brushHeightmap != null)
            {
                GUILayout.Box(brushHeightmap);
            }
            if(brushHeightmapMasked != null)
            {
                GUILayout.Box(brushHeightmapMasked);
            }
        }
    }

    private void RenderIntoPaintContext(PaintContext paintContext, Texture brushTexture, BrushTransform brushXform)
    {
        // Get the built-in painting Material reference
        Material mat = TerrainPaintUtility.GetBuiltinPaintMaterial();
        // Bind the current brush texture
        mat.SetTexture("_BrushTex", brushTexture);
        // Bind the tool-specific shader properties
        var opacity = Event.current.control ? -m_BrushOpacity : m_BrushOpacity;
        mat.SetVector("_BrushParams", new Vector4(opacity, 0.0f, 0.0f, 0.0f));
        // Setup the material for reading from/writing into the PaintContext texture data. This is a necessary step to setup the correct shader properties for appropriately transforming UVs and sampling textures within the shader
        TerrainPaintUtility.SetupTerrainToolMaterialProperties(paintContext, brushXform, mat);
        // Render into the PaintContext's destinationRenderTexture using the built-in painting Material - the id for the Raise/Lower pass is 0.
        Graphics.Blit(paintContext.sourceRenderTexture, paintContext.destinationRenderTexture, mat, 0);
    }

    // Render Tool previews in the SceneView
    public override void OnRenderBrushPreview(Terrain terrain, IOnSceneGUI editContext)
    {
        // Don't do anything if brushes are disabled.
        if(!brushesEnabled) { return; }

        // Dont render preview if this isnt a Repaint
        if(Event.current.type != EventType.Repaint) { return; }

        // Only do the rest if user mouse hits valid terrain
        if(!editContext.hitValidTerrain) { return; }

        // Get the current BrushTransform under the mouse position relative to the Terrain
        BrushTransform brushXform = TerrainPaintUtility.CalculateBrushTransform(terrain, editContext.raycastHit.textureCoord, m_BrushSize, m_BrushRotation);
        // Get the PaintContext for the current BrushTransform. This has a sourceRenderTexture from which to read existing Terrain texture data.
        PaintContext paintContext = TerrainPaintUtility.BeginPaintHeightmap(terrain, brushXform.GetBrushXYBounds(), 1);
        // Get the built-in Material for rendering Brush Previews
        Material previewMaterial = TerrainPaintUtilityEditor.GetDefaultBrushPreviewMaterial();
        // Render the brush preview for the sourceRenderTexture. This will show up as a projected brush mesh rendered on top of the Terrain
        TerrainPaintUtilityEditor.DrawBrushPreview(paintContext, TerrainBrushPreviewMode.SourceRenderTexture, brushHeightmapMasked, brushXform, previewMaterial, 0);
        // Render changes into the PaintContext destinationRenderTexture
        RenderIntoPaintContext(paintContext, brushHeightmapMasked, brushXform);
        // Restore old render target.
        RenderTexture.active = paintContext.oldRenderTexture;
        // Bind the sourceRenderTexture to the preview Material. This is used to compute deltas in height
        previewMaterial.SetTexture("_HeightmapOrig", paintContext.sourceRenderTexture);
        // Render a procedural mesh displaying the delta/displacement in height from the source Terrain texture data. When modifying Terrain height, this shows how much the next paint operation will alter the Terrain height
        TerrainPaintUtilityEditor.DrawBrushPreview(paintContext, TerrainBrushPreviewMode.DestinationRenderTexture, brushHeightmapMasked, brushXform, previewMaterial, 1);
        // Cleanup resources
        TerrainPaintUtility.ReleaseContextResources(paintContext);
    }

    // Perform painting operations that modify the Terrain texture data
    public override bool OnPaint(Terrain terrain, IOnPaint editContext)
    {
        if(!brushesEnabled) { return false; }

        // Get the current BrushTransform under the mouse position relative to the Terrain
        BrushTransform brushXform = TerrainPaintUtility.CalculateBrushTransform(terrain, editContext.uv, m_BrushSize, m_BrushRotation);
        // Get the PaintContext for the current BrushTransform. This has a sourceRenderTexture from which to read existing Terrain texture data
        // and a destinationRenderTexture into which to write new Terrain texture data
        PaintContext paintContext = TerrainPaintUtility.BeginPaintHeightmap(terrain, brushXform.GetBrushXYBounds());
        // Call the common rendering function used by OnRenderBrushPreview and OnPaint
        RenderIntoPaintContext(paintContext, brushHeightmapMasked, brushXform);
        // Commit the modified PaintContext with a provided string for tracking Undo operations. This function handles Undo and resource cleanup for you
        TerrainPaintUtility.EndPaintHeightmap(paintContext, "Terrain Paint - Raise or Lower Height");

        // Return whether or not Trees and Details should be hidden while painting with this Terrain Tool
        return true;
    }

    private void CalculateUpSampledDimensions()
    {
        int upSampleFactor = (int)upSampleResolution;
        upSampledWidth = modelOutputWidth * upSampleFactor;
        upSampledHeight = modelOutputHeight * upSampleFactor;
    }

    private void CalculateBlendingRadii()
    {
        radius1 = upSampledWidth / 2;
        radius2 = upSampledWidth;
    }

    private float[] GenerateHeightmap()
    {
        if(runtimeModel == null)
        {
            if(modelAsset != null)
            {
                runtimeModel = ModelLoader.Load(modelAsset);
            }
            else
            {
                Debug.LogError("Model asset is null.");
                return null;
            }
        }

        float[] heightmap = new float[upSampledWidth * upSampledHeight];
        using(var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel))
        {
            Tensor input = tensorMathHelper.RandomNormalTensor(
                1, modelOutputWidth, modelOutputHeight, channels
            );
            Tensor reverseDiffusionOutput = diffuser.ReverseDiffusion(
                worker, 
                input, 
                20, 
                modelOutputWidth, 
                modelOutputHeight
            );
            
            if(upSampleResolution != UpSampleResolution._256)
            {
                int upSampleFactor = (int)upSampleResolution;
                heightmap = bicubicUpSampler.BicubicUpSample(reverseDiffusionOutput, upSampleFactor);
            }
            else
            {
                heightmap = reverseDiffusionOutput.ToReadOnlyArray();
            }

            input.Dispose();
            reverseDiffusionOutput.Dispose();
        }

        return heightmap;
    }
}