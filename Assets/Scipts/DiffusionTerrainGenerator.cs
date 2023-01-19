using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class DiffusionTerrainGenerator : MonoBehaviour
{
    [SerializeField] private NNModel modelAsset;
    private Model runtimeModel;

    private const int modelOutputWidth = 256;
    private const int modelOutputHeight = 256;
    private const int modelOutputArea = modelOutputWidth * modelOutputHeight;

    // Todo: tweak.
    private const float maxSignalRate = 0.9f;
    private const float minSignalRate = 0.1f;

    private Single[] GenerateHeightmap(Model model)
    {
        // Using ComputePrecompiled worker type for most efficient computation on GPU.
        // Reference: https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        Tensor output = ReverseDiffusion(worker, 20);
        Single[] outputArray = output.ToReadOnlyArray();

        output.Dispose();
        worker.Dispose();

        return outputArray;
    }

    private Tensor ReverseDiffusion(IWorker worker, int diffusionSteps)
    {
        Tensor initialNoise = RandomNormalTensor(modelOutputWidth, modelOutputHeight);
        float stepSize = 1.0f / diffusionSteps;

        Tensor nextNoisyImages = initialNoise;
        Tensor predictedImages = new Tensor(1, modelOutputWidth, modelOutputHeight, 1);
        for(int i = 0; i < diffusionSteps; i++)
        {
            Tensor noisyImages = nextNoisyImages;

            float[] diffusionTimes = {1.0f};
            float[,] rates = DiffusionSchedule(diffusionTimes);

            Tensor noiseRatesSquared = new Tensor(1, diffusionTimes.Length);
            for(int k = 0; k < diffusionTimes.Length; k++)
            {
                float noiseRate = rates[i, 0];
                noiseRatesSquared[i] = Mathf.Pow(noiseRate, 2);
            }

            //worker.Execute({noisyImage, noiseRatesSquared});

            Tensor predictedNoises = worker.PeekOutput();
            for(int k = 0; k < diffusionTimes.Length; k++)
            {
                float noiseRate = rates[i, 0];
                float signalRate = rates[i, 1];

                predictedImages[i] = (noisyImages[i] - noiseRate * predictedNoises[i]) / signalRate;
            }

            float[] nextDiffusionTimes = new float[diffusionTimes.Length];
            for(int k = 0; k < diffusionTimes.Length; k++)
            {
                nextDiffusionTimes[i] = diffusionTimes[i] - stepSize;
            }

            float[,] nextRates = DiffusionSchedule(nextDiffusionTimes);
            for(int k = 0; k < diffusionTimes.Length; k++)
            {
                float nextNoiseRate = nextRates[i, 0];
                float nextSignalRate = nextRates[i, 1];

                nextNoisyImages[i] = nextSignalRate * predictedImages[i] + nextNoiseRate * predictedNoises[i];
            }

            noisyImages.Dispose();
            noiseRatesSquared.Dispose();
            predictedNoises.Dispose();
            predictedImages.Dispose();
        }

        initialNoise.Dispose();
        nextNoisyImages.Dispose();
        return predictedImages;   
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