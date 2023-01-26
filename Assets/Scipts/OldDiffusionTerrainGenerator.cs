using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class OldDiffusionTerrainGenerator : MonoBehaviour
{
    [SerializeField] private NNModel modelAsset;
    private Model runtimeModel;
    [SerializeField] Terrain terrain;
    [SerializeField] private float heightMultiplier = 10.0f;

    [SerializeField] private int cutoffStep;

    private const int modelOutputWidth = 256;
    private const int modelOutputHeight = 256;
    private const int modelOutputArea = modelOutputWidth * modelOutputHeight;

    private const float maxSignalRate = 0.95f;
    private const float minSignalRate = 0.02f;

    private void Start()
    {
        terrain.terrainData.heightmapResolution = 256;
        runtimeModel = ModelLoader.Load(modelAsset);
        Single[] heightmap = GenerateHeightmap(runtimeModel);
        SetTerrainHeights(heightmap);
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            Single[] heightmap = GenerateHeightmap(runtimeModel);
            SetTerrainHeights(heightmap);
        }
    }

    private Single[] GenerateHeightmap(Model model)
    {
        // Using ComputePrecompiled worker type for most efficient computation on GPU.
        // Reference: https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        Tensor output = ReverseDiffusion(worker, 20, 1);
        Single[] outputArray = output.ToReadOnlyArray();

        output.Dispose();
        worker.Dispose();

        return outputArray;
    }

    private void SetTerrainHeights(Single[] heightmap)
    {
        Debug.Log("Heightmap:");
        for(int i = 0; i < 10; i++)
        {
            Debug.Log(heightmap[i]);
        }
        float[,] newHeightmap = new float[modelOutputWidth, modelOutputHeight];
        for(int i = 0; i < modelOutputArea; i++)
        {
            int x = (int)(i % modelOutputWidth);
            int y = (int)Math.Floor((double)(i / modelOutputWidth));
            newHeightmap[x, y] = (float)heightmap[i] * heightMultiplier;
        }

        terrain.terrainData.SetHeights(0, 0, newHeightmap);
    }

    private Tensor ReverseDiffusion(IWorker worker, int diffusionSteps, int batchSize)
    {
        Tensor initialNoise = RandomNormalTensor(modelOutputWidth, modelOutputHeight);
        //DisplayFirstTen(initialNoise, "initialNoise");

        float stepSize = 1.0f / diffusionSteps;
        Debug.Log("stepSize: " + stepSize);

        Tensor nextNoisyImages = initialNoise;
        //DisplayFirstTen(nextNoisyImages, "nextNoisyImages");
        Tensor predictedImages = new Tensor(1, modelOutputWidth, modelOutputHeight, 1);
        for(int i = 0; i < diffusionSteps; i++)
        {
            Tensor noisyImages = nextNoisyImages;

            float[] diffusionTimes = {1.0f - i * stepSize};
            //Debug.Log("diffusionTimes: " + diffusionTimes[0]);
            float[,] rates = DiffusionSchedule(diffusionTimes);
            //Debug.Log("rates: " + rates[0, 0] + ", " + rates[0, 1]);

            Tensor noiseRatesSquared = new Tensor(1, diffusionTimes.Length);
            for(int k = 0; k < diffusionTimes.Length; k++)
            {
                float noiseRate = rates[k, 0];
                noiseRatesSquared[k] = Mathf.Pow(noiseRate, 2);
            }

            IDictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
            inputs.Add("input_1", noisyImages);
            inputs.Add("input_2", noiseRatesSquared);
            worker.Execute(inputs);

            Tensor predictedNoises = worker.PeekOutput();
            //DisplayFirstTen(predictedNoises, "predictedNoises" + i);
            for(int k = 0; k < batchSize; k++)
            {
                float noiseRate = rates[k, 0];
                float signalRate = rates[k, 1];

                for(int j = 0; j < modelOutputArea; j++)
                {
                    predictedImages[k + j] = (noisyImages[k + j] - noiseRate * predictedNoises[k + j]) / signalRate;
                }
            }

            float[] nextDiffusionTimes = new float[diffusionTimes.Length];
            for(int k = 0; k < batchSize; k++)
            {
                nextDiffusionTimes[k] = diffusionTimes[k] - stepSize;
            }

            float[,] nextRates = DiffusionSchedule(nextDiffusionTimes);
            for(int k = 0; k < batchSize; k++)
            {
                float nextNoiseRate = nextRates[k, 0];
                float nextSignalRate = nextRates[k, 1];

                for(int j = 0; j < modelOutputArea; j++)
                {
                    nextNoisyImages[k + j] = nextSignalRate * predictedImages[k + j] + nextNoiseRate * predictedNoises[k + j];
                }
            }

            noisyImages.Dispose();
            noiseRatesSquared.Dispose();
            predictedNoises.Dispose();
            DisplayFirstTen(predictedImages, "predictedImages" + i);
            if(i == cutoffStep)
            {
                initialNoise.Dispose();
                nextNoisyImages.Dispose();
                return predictedImages;
            }
        }
        //DisplayFirstTen(predictedImages, "predictedImagesFinal");
        initialNoise.Dispose();
        nextNoisyImages.Dispose();
        return predictedImages;   
    }

    private void DisplayFirstTen(Tensor tensor, String name)
    {
        Debug.Log(name + ":");
        for(int i = 0; i < 10; i++)
        {
            Debug.Log(tensor[i]);
        }
        Debug.Log("");
    }

    private float[,] DiffusionSchedule(float[] diffusionTimes)
    {   
        // Diffusion times -> angles.
        float startAngle = Mathf.Acos(maxSignalRate);
        float endAngle = Mathf.Acos(minSignalRate);

        float[] diffusionAngles = new Single[diffusionTimes.Length];
        for(int i = 0; i < diffusionTimes.Length; i++)
        {
            diffusionAngles[i] = startAngle + diffusionTimes[i] * (endAngle - startAngle);
        }

        float[,] rates = new float[diffusionTimes.Length, 2];
        for(int i = 0; i < diffusionTimes.Length; i++)
        {
            float signalRate = Mathf.Cos(diffusionAngles[i]);
            float noiseRate = Mathf.Sin(diffusionAngles[i]); 
            rates[i, 0] = noiseRate;
            rates[i, 1] = signalRate;
        }
        return rates;
    }

    private Tensor RandomNormalTensor(int width, int height)
    {
        Tensor tensor = new Tensor(1, width, height, 1);
        System.Random random = new System.Random();
        for(int i = 0; i < width * height; i++)
        {
            // Box-Muller transform.
            // Reference: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
            double mean = 0.0f;
            double stdDev = 1.0f;
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randomStdNormal = Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2.0f * Math.PI * u2);
            double randomNormal = mean + stdDev * randomStdNormal;
            tensor[i] = (float)randomNormal;
        }
        return tensor;
    }
}