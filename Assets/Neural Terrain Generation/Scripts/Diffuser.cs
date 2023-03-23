using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    public class Diffuser
    {
        private TensorMathHelper tensorMathHelper = new TensorMathHelper();

        private Tensor[] DiffusionSchedule(
            float[] diffusionTimes, 
            float minSignalRate, 
            float maxSignalRate
        )
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

        private IDictionary<string, Tensor> PackageInputs(
            Tensor noisyImages, 
            Tensor noiseRatesSquared
        )
        {
            IDictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
            inputs.Add("input_1", noisyImages);
            inputs.Add("input_2", noiseRatesSquared);
            return inputs;
        }

        public Tensor ReverseDiffusion(
            IWorker worker,
            Tensor initialNoise, 
            int diffusionSteps,
            int modelOutputWidth,
            int modelOutputHeight,
            int channels = 1,
            int startingStep = 0, 
            int batchSize = 1,
            float minSignalRate = 0.02f,
            float maxSignalRate = 0.9f
        )
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
                Tensor[] rates = DiffusionSchedule(diffusionTimes, minSignalRate, maxSignalRate);
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

                Tensor[] nextRates = DiffusionSchedule(nextDiffusionTimes, minSignalRate, maxSignalRate);
                Tensor nextNoiseRates = nextRates[0];
                Tensor nextSignalRates = nextRates[1];

                // predictedImages * nextSignalRate
                Tensor scaledPredictedImages = tensorMathHelper.ScaleTensorBatches(
                    predictedImages, 
                    nextSignalRates
                );

                // predictedNoises * nextNoiseRate
                Tensor scaledPredictedNoises2 = tensorMathHelper.ScaleTensorBatches(
                    predictedNoises, 
                    nextNoiseRates
                );

                // predictedImages * nextSignalRate + predictedNoises * nextNoiseRate
                nextNoisyImages = tensorMathHelper.AddTensor(
                    scaledPredictedImages, 
                    scaledPredictedNoises2
                );
                
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
    }
}
