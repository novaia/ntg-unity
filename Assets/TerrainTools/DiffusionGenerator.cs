using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEditor;
using UnityEditor.TerrainTools;

public class DiffusionGenerator : TerrainPaintTool<DiffusionGenerator>
{
    // General.
    private TensorMathHelper tensorMathHelper = new TensorMathHelper();
    private int modelOutputWidth = 256;
    private int modelOutputHeight = 256;
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
    private int upSampleFactor = 2;
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
        modelAsset = (NNModel)EditorGUILayout.ObjectField("Model Asset", modelAsset, typeof(NNModel), false);
        modelOutputWidth = EditorGUILayout.IntField("Model Output Width", modelOutputWidth);
        modelOutputHeight = EditorGUILayout.IntField("Model Output Height", modelOutputHeight);
        heightMultiplier = EditorGUILayout.FloatField("Height Multiplier", heightMultiplier);
        upSampleFactor = EditorGUILayout.IntField("UpSample Factor", upSampleFactor);
        upSampleMode = (UpSampleMode)EditorGUILayout.EnumPopup("UpSample Mode", upSampleMode);
        upSampleResolution = (UpSampleResolution)EditorGUILayout.EnumPopup("UpSample Resolution", upSampleResolution);

        if(GUILayout.Button("Generate Terrain From Scratch"))
        {
            float[] heightmap = GenerateHeightmap();
            int width = modelOutputWidth * (int)upSampleResolution;
            int height = modelOutputHeight * (int)upSampleResolution;
            SetTerrainHeights(terrain, heightmap, width, height);
        }

        bValue = EditorGUILayout.FloatField("B Value", bValue);
        keepNeighborHeights = EditorGUILayout.Toggle("Keep Neighbor Heights", keepNeighborHeights);

        if(GUILayout.Button("Blend With Neighbors"))
        {
            BlendAllNeighbors(terrain);
        }
    }

    private int[] CalculateUpSampledDimensions()
    {
        int upSampleFactor = (int)upSampleResolution;
        int[] dimensions = new int[] 
        {
            modelOutputWidth * upSampleFactor,
            modelOutputHeight * upSampleFactor,
        };
        return dimensions;
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

        int[] heightmapDimensions = CalculateUpSampledDimensions();
        float[] heightmap = new float[heightmapDimensions[0] * heightmapDimensions[1]];
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
        float [,] neighborHeightmapArray = neighbor.terrainData.GetHeights(
            0, 0, modelOutputWidth, modelOutputHeight
        );
        Tensor neighborHeightmap = tensorMathHelper.TwoDimensionalArrayToTensor(neighborHeightmapArray);

        Tensor localGradient = new Tensor(1, 256, 256, 1);
        for(int x = 0; x < 256; x++)
        {
            for(int y = 0; y < 256; y++)
            {
                localGradient[0, x, y, 0] = gradient[0, yOffset + y, xOffset + x, 0];
            }
        }
        Tensor scaledMirror = tensorMathHelper.MultiplyTensors(localGradient, mirror);

        if(keepNeighborHeights)
        {
            Tensor OnesTensor = tensorMathHelper.PopulatedTensor(1.0f, modelOutputWidth, modelOutputHeight);
            Tensor inverseLocalGradient = tensorMathHelper.SubtractTensor(OnesTensor, localGradient);
            Tensor scaledNeighbor = tensorMathHelper.MultiplyTensors(inverseLocalGradient, neighborHeightmap);
            Tensor blended = tensorMathHelper.AddTensor(scaledMirror, scaledNeighbor);

            SetTerrainHeights(neighbor, blended.ToReadOnlyArray(), modelOutputWidth, modelOutputHeight, false);

            OnesTensor.Dispose();
            inverseLocalGradient.Dispose();
            scaledNeighbor.Dispose();
            blended.Dispose();
        }
        else
        {
            SetTerrainHeights(neighbor, scaledMirror.ToReadOnlyArray(), modelOutputWidth, modelOutputHeight, false);
        }

        neighborHeightmap.Dispose();
        localGradient.Dispose();
        scaledMirror.Dispose();
    }

    private void BlendAllNeighbors(Terrain terrain)
    {
        float[,] heightmap = terrain.terrainData.GetHeights(0, 0, 256, 256);
        Tensor heightmapTensor = tensorMathHelper.TwoDimensionalArrayToTensor(heightmap);
        Tensor horizontalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, false, true);
        Tensor verticalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, false);
        Tensor bothMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, true);

        Debug.Log("Mirror Width: " + horizontalMirror.width);

        Tensor gradient = new Tensor(1, 256 * 3, 256 * 3, 1);
        Vector2 center = new Vector2(radius1 + radius2, radius1 + radius2);
        for(int x = 0; x < 256 * 3; x++)
        {
            for(int y = 0; y < 256 * 3; y++)
            {
                float distance = Vector2.Distance(new Vector2(x, y), center);
                if(distance < radius1)
                {
                    gradient[0, x, y, 0] = 1.0f;
                }
                else
                {
                    float gradientValue = (-1.0f / 128.0f) * distance + bValue;
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

        Tensor gradientTest = tensorMathHelper.GradientTensor(0.0f, 0.0f, 1.0f, 0.0f, 256, 256);
        
        Terrain topLeftNeighbor = null;
        Terrain bottomLeftNeighbor = null;
        Terrain topRightNeighbor = null;
        Terrain bottomRightNeighbor = null;

        // Split gradient in 8 parts.
        Terrain leftNeighbor = terrain.leftNeighbor;
        if(leftNeighbor != null)
        {
            BlendSingleNeighbor(leftNeighbor, horizontalMirror, gradient, 0, 256);

            topLeftNeighbor = leftNeighbor.topNeighbor;
            bottomLeftNeighbor = leftNeighbor.bottomNeighbor;
        }

        Terrain rightNeighbor = terrain.rightNeighbor;
        if(rightNeighbor != null)
        {
            BlendSingleNeighbor(rightNeighbor, horizontalMirror, gradient, 512, 256);

            topRightNeighbor = rightNeighbor.topNeighbor;
            bottomRightNeighbor = rightNeighbor.bottomNeighbor;
        }

        Terrain topNeighbor = terrain.topNeighbor;
        if(topNeighbor != null)
        {
            BlendSingleNeighbor(topNeighbor, verticalMirror, gradient, 256, 512);
        }

        Terrain bottomNeighbor = terrain.bottomNeighbor;
        if(bottomNeighbor != null)
        {
            BlendSingleNeighbor(bottomNeighbor, verticalMirror, gradient, 256, 0);
        }
        
        if(topLeftNeighbor != null)
        {
            BlendSingleNeighbor(topLeftNeighbor, bothMirror, gradient, 0, 512);
        }

        if(bottomLeftNeighbor != null)
        {
            BlendSingleNeighbor(bottomLeftNeighbor, bothMirror, gradient, 0, 0);
        }

        if(topRightNeighbor != null)
        {
            BlendSingleNeighbor(topRightNeighbor, bothMirror, gradient, 512, 512);
        }

        if(bottomRightNeighbor != null)
        {
            BlendSingleNeighbor(bottomRightNeighbor, bothMirror, gradient, 512, 0);
        }
    }


    public void SetTerrainHeights(Terrain terrain, float[] heightmap, int width, int height, bool scale = true)
    {
        Debug.Log("Width: " + width);
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
        Tensor upSampledWidth = new Tensor(1, original.height, original.width * factor, 1);
        Tensor upSampled = new Tensor(1, original.height * factor, original.width * factor, 1);

        float p0 = 0;
        float p1 = 0;
        float p2 = 0;
        float p3 = 0;

        // Up sample width.
        for(int x = 1; x < original.width - 2; x++)
        {
            for(int y = 0; y < original.height; y++)
            {
                p0 = original[0, y, x - 1, 0];
                p1 = original[0, y, x, 0];
                p2 = original[0, y, x + 1, 0];
                p3 = original[0, y, x + 2, 0];

                float[] samples = SampleCubic(factor, p0, p1, p2, p3);
                for(int i = 0; i < factor - 1; i++)
                {
                    upSampledWidth[0, y, x * factor + i + 1, 0] = samples[i];
                }
                upSampledWidth [0, y, x * factor, 0] = p1;
            }
        }

        // Up sample height.
        for(int x = 0; x < upSampledWidth.width; x++)
        {
            for(int y = 1; y < original.height - 2; y++)
            {
                p0 = upSampledWidth[0, y - 1, x, 0];
                p1 = upSampledWidth[0, y, x, 0];
                p2 = upSampledWidth[0, y + 1, x, 0];
                p3 = upSampledWidth[0, y + 2, x, 0];

                float[] samples = SampleCubic(factor, p0, p1, p2, p3);
                for(int i = 0; i < factor - 1; i++)
                {
                    upSampled[0, y * factor + i + 1, x, 0] = samples[i];
                }
                upSampled[0, y * factor, x, 0] = p1;
            }
        }

        float[] upSampledArray = upSampled.ToReadOnlyArray();
        upSampledWidth.Dispose();
        upSampled.Dispose();
        return upSampledArray;
    }
}