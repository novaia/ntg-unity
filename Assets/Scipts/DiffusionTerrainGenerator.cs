using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Unity.Barracuda;

public class DiffusionTerrainGenerator : BaseTerrainGenerator
{
    private const float maxSignalRate = 0.95f;
    private const float minSignalRate = 0.02f;

    [SerializeField] protected float existingHeightmapWeight;
    [SerializeField] protected int diffusionIterationsFromScratch = 20;
    [SerializeField] protected int diffusionIterationsFromExisting = 20;
    public Tensor randomNormalTensorForDisplay;
    protected float[,] flatHeights;

    public override void Setup()
    {
        base.Setup();
        randomNormalTensorForDisplay = tensorMathHelper.RandomNormalTensor(0, 
                                                                           modelOutputWidth, 
                                                                           modelOutputHeight, 
                                                                           channels);
        flatHeights = terrain.terrainData.GetHeights(0, 0, 
                                                     modelOutputWidth, 
                                                     modelOutputHeight);
        for(int x = 0; x < modelOutputWidth; x++)
        {
            for(int y = 0; y < modelOutputHeight; y++)
            {
                flatHeights[x, y] = 0f;
            }
        }
    }

    public void ClearTerrain()
    {
        terrain.terrainData.SetHeights(0, 0, flatHeights);
    }

    public Texture2D GetTerrainHeightmapAsTexture()
    {
        float[,] heightmap = terrain.terrainData.GetHeights(0, 0, 
                                                            modelOutputWidth, 
                                                            modelOutputHeight);
        Texture2D heightmapTexture = new Texture2D(modelOutputWidth, 
                                                   modelOutputHeight);
        
        float noiseWeight = 1 - existingHeightmapWeight;

        for(int x = 0; x < modelOutputWidth; x++)
        {
            for(int y = 0; y < modelOutputHeight; y++)
            {
                float weightedHeightmapValue = heightmap[x, y] * existingHeightmapWeight;
                float weightedNoiseValue = randomNormalTensorForDisplay[x + y * modelOutputWidth] * noiseWeight;
                float colorValue = (weightedHeightmapValue + weightedNoiseValue) * 2;
                Color color = new Color(colorValue, colorValue, colorValue);
                heightmapTexture.SetPixel(x, y, color);
            }
        }
        heightmapTexture.Apply();

        return heightmapTexture;
    }

    public float[] GenerateHeightmapFromScratch()
    {
        Tensor initialNoise = tensorMathHelper.RandomNormalTensor(0, 
                                                                  modelOutputWidth, 
                                                                  modelOutputHeight, 
                                                                  channels);

        object [] args = new object[] {diffusionIterationsFromScratch, 1, initialNoise};
        float[] heightmap = GenerateHeightmap(runtimeModel, 
                                              new WorkerExecuter(ReverseDiffusion),
                                              args);
        return heightmap;
    }

    public float[] GenerateHeightmapFromExisting()
    {
        if(terrain.terrainData.heightmapResolution != modelOutputWidth
           || terrain.terrainData.heightmapResolution != modelOutputHeight)
        {
            //Debug.LogError("Terrain heightmap size must match model output size.");
            //return null;
        }

        float[,] existingHeightmap = terrain.terrainData.GetHeights(0, 0, 
                                                                    modelOutputWidth, 
                                                                    modelOutputHeight);
        float maxHeight = existingHeightmap[0, 0];
        float minHeight = existingHeightmap[0, 0];

        Tensor normalizedExistingHeightmap = new Tensor(0, modelOutputHeight, 
                                                        modelOutputWidth, channels);
        for(int x = 0; x < modelOutputWidth; x++)
        {
            for(int y = 0; y < modelOutputHeight; y++)
            {
                if(existingHeightmap[x, y] > maxHeight)
                {
                    maxHeight = existingHeightmap[x, y];
                }
                else if (existingHeightmap[x, y] < minHeight)
                {
                    minHeight = existingHeightmap[x, y];
                }
            }
        }

        /*float minMaxDifference = maxHeight - minHeight;
        for(int x = 0; x < modelOutputWidth; x++)
        {
            for(int y = 0; y < modelOutputHeight; y++)
            {
                float normalizedValue = (existingHeightmap[x, y] - minHeight) / minMaxDifference;
                normalizedExistingHeightmap[0, y, x, 0] = normalizedValue - 0.5f;
            }
        }*/
        if(maxHeight != 0)
        {
            for(int x = 0; x < modelOutputWidth; x++)
            {
                for(int y = 0; y < modelOutputHeight; y++)
                {
                    float constrainedValue = existingHeightmap[x, y] / maxHeight;
                    normalizedExistingHeightmap[0, y, x, 0] = constrainedValue;
                    Debug.Log(constrainedValue);
                }
            }
        }
        
        Tensor initialNoise = tensorMathHelper.RandomNormalTensor(0, 
                                                                  modelOutputWidth, 
                                                                  modelOutputHeight, 
                                                                  channels);
        Tensor scaledInitialNoise = tensorMathHelper.ScaleTensor(initialNoise, 
                                                                 1 - existingHeightmapWeight);
        Tensor scaledExistingHeightmap = tensorMathHelper.ScaleTensor(normalizedExistingHeightmap, 
                                                                      existingHeightmapWeight);
        Tensor inputTensor = tensorMathHelper.AddTensor(scaledExistingHeightmap, scaledInitialNoise);

        object[] args = new object[] {diffusionIterationsFromExisting, 1, inputTensor};
        float[] heightmap = GenerateHeightmap(runtimeModel, 
                                              new WorkerExecuter(ReverseDiffusion),
                                              args);
        
        // Make minimum height 0.
        minHeight = heightmap[0];
        int minIndex = 0;
        for(int i = 0; i < modelOutputArea; i++)
        {
            if(heightmap[i] < minHeight)
            {
                minHeight = heightmap[i];
                minIndex = i;
            }
        }
        for(int i = 0; i < modelOutputArea; i++)
        {
            heightmap[i] -= minHeight;
        }

        return heightmap;
    } 

    protected Tensor ReverseDiffusion(IWorker worker, params object[] args)
    {
        int diffusionSteps = (int)args[0];
        int batchSize = (int)args[1];
        Tensor initialNoise = (Tensor)args[2];

        float stepSize = 1.0f / diffusionSteps;
        Tensor nextNoisyImages = initialNoise;
        Tensor predictedImages = new Tensor(batchSize, modelOutputWidth, 
                                            modelOutputHeight, channels);

        for(int step = 0; step < diffusionSteps; step++)
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
            Tensor scaledPredictedNoises = tensorMathHelper.ScaleTensorBatches(predictedNoises, 
                                                                               noiseRates);
            // noisyImages - predictedNoises * noiseRate
            Tensor difference = tensorMathHelper.SubtractTensor(noisyImages, 
                                                                scaledPredictedNoises);
            // (noisyImages - predictedNoises * noiseRate) / signalRate
            predictedImages = tensorMathHelper.ScaleTensorBatches(difference, 
                                                                  signalRates,
                                                                  true);

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
}
