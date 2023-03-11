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
    
    // Diffusion.
    private const float maxSignalRate = 0.9f;
    private const float minSignalRate = 0.02f;

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

    // Brushes.
    private bool brushesEnabled = false;
    private Texture2D brushTexture1;
    private Texture2D brushTexture2;
    private float m_BrushOpacity = 0.1f;
    private float m_BrushSize = 25f;
    private float m_BrushRotation = 0f;

    public override string GetName()
    {
        return "NTG/" + "Diffusion Generator";
    }

    public override string GetDescription()
    {
        return "Diffusion based neural terrain generator.";
    }

    public override void OnInspectorGUI(Terrain terrain, IOnInspectorGUI editContext)
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
            editContext.ShowBrushesGUI(5, BrushGUIEditFlags.Select);
            m_BrushOpacity = EditorGUILayout.Slider("Opacity", m_BrushOpacity, 0, 1);
            m_BrushSize = EditorGUILayout.Slider("Size", m_BrushSize, .001f, 2000f);
            m_BrushRotation = EditorGUILayout.Slider("Rotation", m_BrushRotation, 0, 360);
            if(GUILayout.Button("Generate Brush Heighmap"))
            {
                float[] brushHeightmap = GenerateHeightmap();
                Color[] colorBrushHeightmap = new Color[brushHeightmap.Length];
                for(int i = 0; i < brushHeightmap.Length; i++)
                {
                    colorBrushHeightmap[i] = new Color(brushHeightmap[i], brushHeightmap[i], brushHeightmap[i]);
                }
                brushTexture1 = new Texture2D(upSampledWidth, upSampledHeight);
                brushTexture1.SetPixels(0, 0, upSampledWidth, upSampledHeight, colorBrushHeightmap);
                brushTexture1.Apply();
            }
            GUILayout.Box(brushTexture1);
        }

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
            SetTerrainHeights(terrain, heightmap, upSampledWidth, upSampledHeight);
        }

        bValue = EditorGUILayout.FloatField("B Value", bValue);
        keepNeighborHeights = EditorGUILayout.Toggle("Keep Neighbor Heights", keepNeighborHeights);

        if(GUILayout.Button("Blend With Neighbors"))
        {
            BlendAllNeighbors(terrain);
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
        TerrainPaintUtilityEditor.DrawBrushPreview(paintContext, TerrainBrushPreviewMode.SourceRenderTexture, brushTexture1, brushXform, previewMaterial, 0);
        // Render changes into the PaintContext destinationRenderTexture
        RenderIntoPaintContext(paintContext, brushTexture1, brushXform);
        // Restore old render target.
        RenderTexture.active = paintContext.oldRenderTexture;
        // Bind the sourceRenderTexture to the preview Material. This is used to compute deltas in height
        previewMaterial.SetTexture("_HeightmapOrig", paintContext.sourceRenderTexture);
        // Render a procedural mesh displaying the delta/displacement in height from the source Terrain texture data. When modifying Terrain height, this shows how much the next paint operation will alter the Terrain height
        TerrainPaintUtilityEditor.DrawBrushPreview(paintContext, TerrainBrushPreviewMode.DestinationRenderTexture, brushTexture1, brushXform, previewMaterial, 1);
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
        RenderIntoPaintContext(paintContext, brushTexture1, brushXform);
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
            Tensor reverseDiffusionOutput = ReverseDiffusion(worker, input, 20);
            
            if(upSampleResolution != UpSampleResolution._256)
            {
                int upSampleFactor = (int)upSampleResolution;
                heightmap = BicubicUpSample(reverseDiffusionOutput, upSampleFactor);
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

    private Tensor ReverseDiffusion(IWorker worker,
                                    Tensor initialNoise, 
                                    int diffusionSteps, 
                                    int startingStep = 0, 
                                    int batchSize = 1)
    {
        float stepSize = 1.0f / diffusionSteps;

        Tensor nextNoisyImages = initialNoise;
        Tensor predictedImages = new Tensor(
            batchSize, 
            modelOutputWidth, 
            modelOutputHeight, 
            channels
        );

        for(int step = startingStep; step < diffusionSteps; step++)
        {
            Tensor noisyImages = nextNoisyImages;

            float[] diffusionTimes = {1.0f - stepSize * step};
            Tensor[] rates = DiffusionSchedule(diffusionTimes);
            Tensor noiseRates = rates[0];
            Tensor signalRates = rates[1];
            Tensor noiseRatesSquared = tensorMathHelper.RaiseTensorToPower(noiseRates, 2);
            
            worker.Execute(PackageInputs(noisyImages, noiseRatesSquared));
            Tensor predictedNoises = worker.PeekOutput();

            // Calculate predicted images.
            // predictedNoises * noiseRate
            Tensor scaledPredictedNoises = tensorMathHelper.ScaleTensorBatches(
                predictedNoises, noiseRates
            );

            // noisyImages - predictedNoises * noiseRate
            Tensor difference = tensorMathHelper.SubtractTensor(
                noisyImages, scaledPredictedNoises
            );

            // (noisyImages - predictedNoises * noiseRate) / signalRate
            predictedImages = tensorMathHelper.ScaleTensorBatches(
                difference, signalRates, true
            );

            scaledPredictedNoises.Dispose();
            difference.Dispose();

            // Setup for next step.
            float[] nextDiffusionTimes = new float[batchSize];
            for(int i = 0; i < batchSize; i++)
            {
                nextDiffusionTimes[i] = diffusionTimes[i] - stepSize;
            }

            Tensor[] nextRates = DiffusionSchedule(nextDiffusionTimes);
            Tensor nextNoiseRates = nextRates[0];
            Tensor nextSignalRates = nextRates[1];

            // predictedImages * nextSignalRate
            Tensor scaledPredictedImages = tensorMathHelper.ScaleTensorBatches(predictedImages, 
                                                                               nextSignalRates);
            // predictedNoises * nextNoiseRate
            Tensor scaledPredictedNoises2 = tensorMathHelper.ScaleTensorBatches(predictedNoises, 
                                                                                nextNoiseRates);
            // predictedImages * nextSignalRate + predictedNoises * nextNoiseRate
            nextNoisyImages = tensorMathHelper.AddTensor(scaledPredictedImages, 
                                                         scaledPredictedNoises2);
            
            // Dispose of tensors.
            scaledPredictedImages.Dispose();
            scaledPredictedNoises2.Dispose();
            nextNoiseRates.Dispose();
            nextSignalRates.Dispose();
            noiseRates.Dispose();
            signalRates.Dispose();
            noiseRatesSquared.Dispose();
            predictedNoises.Dispose();
            noisyImages.Dispose();
        }

        // Dispose of tensors.
        initialNoise.Dispose();
        nextNoisyImages.Dispose();

        return predictedImages;
    }

    private Tensor[] DiffusionSchedule(float[] diffusionTimes)
    {   
        Tensor noiseRates = new Tensor(1, diffusionTimes.Length);
        Tensor signalRates = new Tensor(1, diffusionTimes.Length);

        float startAngle = Mathf.Acos(maxSignalRate);
        float endAngle = Mathf.Acos(minSignalRate);

        float[] diffusionAngles = new float[diffusionTimes.Length];
        for(int i = 0; i < diffusionTimes.Length; i++)
        {
            diffusionAngles[i] = startAngle + diffusionTimes[i] * (endAngle - startAngle);
        }

        for(int i = 0; i < diffusionTimes.Length; i++)
        {
            noiseRates[i] = Mathf.Sin(diffusionAngles[i]);
            signalRates[i] = Mathf.Cos(diffusionAngles[i]); 
        }
        return new Tensor[] {noiseRates, signalRates};
    }

    protected IDictionary<string, Tensor> PackageInputs(Tensor noisyImages, 
                                                        Tensor noiseRatesSquared)
    {
        IDictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
        inputs.Add("input_1", noisyImages);
        inputs.Add("input_2", noiseRatesSquared);
        return inputs;
    }

    private void BlendSingleNeighbor(Terrain neighbor, 
                                     Tensor mirror, 
                                     Tensor gradient, 
                                     int xOffset, 
                                     int yOffset)
    {

        Tensor localGradient = new Tensor(1, upSampledHeight, upSampledWidth, 1);
        for(int x = 0; x < upSampledWidth; x++)
        {
            for(int y = 0; y < upSampledHeight; y++)
            {
                localGradient[0, x, y, 0] = gradient[0, yOffset + y, xOffset + x, 0];
            }
        }
        Tensor scaledMirror = tensorMathHelper.MultiplyTensors(localGradient, mirror);

        if(keepNeighborHeights)
        {
            float [,] neighborHeightmapArray = neighbor.terrainData.GetHeights(
                0, 0, upSampledWidth, upSampledHeight
            );
            Tensor neighborHeightmap = tensorMathHelper.TwoDimensionalArrayToTensor(neighborHeightmapArray);
            Tensor OnesTensor = tensorMathHelper.PopulatedTensor(1.0f, modelOutputWidth, modelOutputHeight);
            Tensor inverseLocalGradient = tensorMathHelper.SubtractTensor(OnesTensor, localGradient);
            Tensor scaledNeighbor = tensorMathHelper.MultiplyTensors(inverseLocalGradient, neighborHeightmap);
            Tensor blended = tensorMathHelper.AddTensor(scaledMirror, scaledNeighbor);

            SetTerrainHeights(neighbor, blended.ToReadOnlyArray(), upSampledWidth, upSampledHeight, false);

            neighborHeightmap.Dispose();
            OnesTensor.Dispose();
            inverseLocalGradient.Dispose();
            scaledNeighbor.Dispose();
            blended.Dispose();
        }
        else
        {
            SetTerrainHeights(neighbor, scaledMirror.ToReadOnlyArray(), upSampledWidth, upSampledHeight, false);
        }
 
        localGradient.Dispose();
        scaledMirror.Dispose();
    }

    private void BlendAllNeighbors(Terrain terrain)
    {
        float[,] heightmap = terrain.terrainData.GetHeights(0, 0, upSampledWidth, upSampledHeight);
        Tensor heightmapTensor = tensorMathHelper.TwoDimensionalArrayToTensor(heightmap);
        Tensor horizontalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, false, true);
        Tensor verticalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, false);
        Tensor bothMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, true);

        Tensor gradient = new Tensor(1, upSampledWidth * 3, upSampledHeight * 3, 1);
        Vector2 center = new Vector2(radius1 + radius2, radius1 + radius2);
        for(int x = 0; x < upSampledWidth * 3; x++)
        {
            for(int y = 0; y < upSampledHeight * 3; y++)
            {
                float distance = Vector2.Distance(new Vector2(x, y), center);
                if(distance < radius1)
                {
                    gradient[0, x, y, 0] = 1.0f;
                }
                else
                {
                    float gradientValue = (-1.0f / radius1) * distance + bValue;
                    if(gradientValue > 1.0f)
                    {
                        gradient[0, x, y, 0] = 1.0f;
                    }
                    else
                    {
                        gradient[0, x, y, 0] = gradientValue;
                    }
                }
            }
        }
        
        Terrain topLeftNeighbor = null;
        Terrain bottomLeftNeighbor = null;
        Terrain topRightNeighbor = null;
        Terrain bottomRightNeighbor = null;

        // Split gradient in 8 parts.
        Terrain leftNeighbor = terrain.leftNeighbor;
        if(leftNeighbor != null)
        {
            BlendSingleNeighbor(leftNeighbor, horizontalMirror, gradient, 0, upSampledHeight);

            topLeftNeighbor = leftNeighbor.topNeighbor;
            bottomLeftNeighbor = leftNeighbor.bottomNeighbor;
        }

        Terrain rightNeighbor = terrain.rightNeighbor;
        if(rightNeighbor != null)
        {
            BlendSingleNeighbor(rightNeighbor, horizontalMirror, gradient, upSampledWidth * 2, upSampledHeight);

            topRightNeighbor = rightNeighbor.topNeighbor;
            bottomRightNeighbor = rightNeighbor.bottomNeighbor;
        }

        Terrain topNeighbor = terrain.topNeighbor;
        if(topNeighbor != null)
        {
            BlendSingleNeighbor(topNeighbor, verticalMirror, gradient, upSampledWidth, upSampledHeight * 2);
        }

        Terrain bottomNeighbor = terrain.bottomNeighbor;
        if(bottomNeighbor != null)
        {
            BlendSingleNeighbor(bottomNeighbor, verticalMirror, gradient, upSampledWidth, 0);
        }
        
        if(topLeftNeighbor != null)
        {
            BlendSingleNeighbor(topLeftNeighbor, bothMirror, gradient, 0, upSampledHeight * 2);
        }

        if(bottomLeftNeighbor != null)
        {
            BlendSingleNeighbor(bottomLeftNeighbor, bothMirror, gradient, 0, 0);
        }

        if(topRightNeighbor != null)
        {
            BlendSingleNeighbor(topRightNeighbor, bothMirror, gradient, upSampledWidth * 2, upSampledHeight * 2);
        }

        if(bottomRightNeighbor != null)
        {
            BlendSingleNeighbor(bottomRightNeighbor, bothMirror, gradient, upSampledWidth * 2, 0);
        }
    }


    public void SetTerrainHeights(Terrain terrain, float[] heightmap, int width, int height, bool scale = true)
    {
        terrain.terrainData.heightmapResolution = width;

        float scaleCoefficient = 1;
        if(scale)
        {
            float maxValue = heightmap[0];
            for(int i = 0; i < heightmap.Length; i++)
            {
                if(heightmap[i] > maxValue)
                {
                    maxValue = heightmap[i];
                }
            }
            scaleCoefficient = (1 / maxValue) * heightMultiplier;
        }

        float[,] newHeightmap = new float[width+1, height+1];
        for(int x = 0; x < width; x++)
        {
            for(int y = 0; y < height; y++)
            {
                newHeightmap[x, y] = heightmap[x + y * width] * scaleCoefficient;
            }
        }

        for(int i = 0; i < width+1; i++)
        {
            newHeightmap[i, height] = newHeightmap[i, height-1];
        }
        for(int i = 0; i < height+1; i++)
        {
            newHeightmap[width, i] = newHeightmap[width-1, i];
        }

        terrain.terrainData.SetHeights(0, 0, newHeightmap);
    }

    private float[] SampleCubic(int num_samples, float p0, float p1, float p2, float p3)
    {
        // f(x) = ax^3 + bx^2 + cx + d
        // f(0) = d = p1
        // f(1) = a + b + c + d = p2
        // f'(x) = 3ax^2 + 2bx + c
        // f'(0) = c = p0 - p1 = tangentStart
        // f'(1) = 3a + 2b + c = p2 - p1 = tangentEnd
        // a + b = p2 - p1 - tangentStart
        // 3a + 2b = tangentEnd - tangentStart
        // (3a + 2b) - 2(a + b) = (tangentEnd - tangentStart) - 2(p2 - p1 - tangentStart)
        // a = (tangentEnd - tangentStart) - 2(p2 - p1 - tangentStart)
        // b = p2 - p1 - tangentStart - a

        float tangentStart = p0 - p1;
        float tangentEnd = p2 - p3;

        float a = (tangentEnd - tangentStart) - 2 * (p2 - p1 - tangentStart);
        float b = p2 - p1 - tangentStart - a;
        float c = tangentStart;
        float d = p1;

        float[] samples = new float[num_samples];
        for(int i = 0; i < num_samples; i++)
        {
            float x = (float)i / (float)num_samples;
            samples[i] = a * Mathf.Pow(x, 3) + b * Mathf.Pow(x, 2) + c * x + d;
        }
        return samples;
    }

    private float[] BicubicUpSample(Tensor original, int factor)
    {
        Tensor upSampledX = new Tensor(1, original.height, original.width * factor, 1);
        Tensor upSampled = new Tensor(1, original.height * factor, original.width * factor, 1);

        float p0 = 0;
        float p1 = 0;
        float p2 = 0;
        float p3 = 0;

        // Up sample width.
        for(int x = 0; x < original.width; x++)
        {
            for(int y = 0; y < original.height; y++)
            {
                // p0
                if(x - 1 < 0)
                {
                    System.Random random = new System.Random();
                    p0 = (float)random.NextDouble();
                }
                else
                {
                    p0 = original[0, y, x - 1, 0];
                }

                // p1
                p1 = original[0, y, x, 0];

                // p2 and p3
                if(x + 2 >= original.width)
                {
                    System.Random random = new System.Random();
                    p2 = (float)random.NextDouble();
                    p3 = (float)random.NextDouble();
                }
                else
                {
                    p2 = original[0, y, x + 1, 0];
                    p3 = original[0, y, x + 2, 0];
                }

                float[] samples = SampleCubic(factor, p0, p1, p2, p3);
                for(int i = 0; i < factor - 1; i++)
                {
                    upSampledX[0, y, x * factor + i + 1, 0] = samples[i];
                }
                upSampledX [0, y, x * factor, 0] = p1;
            }
        }

        // Up sample height.
        for(int x = 0; x < upSampledX.width; x++)
        {
            for(int y = 0; y < original.height; y++)
            {
                // p0
                if(y - 1 < 0)
                {
                    System.Random random = new System.Random();
                    p0 = (float)random.NextDouble();
                }
                else
                {
                    p0 = upSampledX[0, y - 1, x, 0];
                }

                // p1
                p1 = upSampledX[0, y, x, 0];

                // p2 and p3
                if(y + 2 >= original.height)
                {
                    System.Random random = new System.Random();
                    p2 = (float)random.NextDouble();
                    p3 = (float)random.NextDouble();
                }
                else
                {
                    p2 = upSampledX[0, y + 1, x, 0];
                    p3 = upSampledX[0, y + 2, x, 0];
                }

                float[] samples = SampleCubic(factor, p0, p1, p2, p3);
                for(int i = 0; i < factor - 1; i++)
                {
                    upSampled[0, y * factor + i + 1, x, 0] = samples[i];
                }
                upSampled[0, y * factor, x, 0] = p1;
            }
        }

        float[] upSampledArray = upSampled.ToReadOnlyArray();
        upSampledX.Dispose();
        upSampled.Dispose();
        return upSampledArray;
    }
}