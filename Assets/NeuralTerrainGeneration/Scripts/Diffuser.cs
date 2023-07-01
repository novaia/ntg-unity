using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    [System.Serializable]
    public class Diffuser
    {
        public bool IsDisposed { get; private set; }
        public WorkerFactory.Type WorkerType { get; private set; }
        public Model RuntimeModel { get; private set; }
        private IWorker worker;

        private TensorMathHelper tensorMathHelper = new TensorMathHelper();

        public Diffuser(
            WorkerFactory.Type workerType, 
            Model runtimeModel
        )
        {
            InitializeDiffuser(workerType, runtimeModel);
        }

        private void InitializeDiffuser(
            WorkerFactory.Type workerType, Model runtimeModel
        )
        {
            this.WorkerType = workerType;
            this.RuntimeModel = runtimeModel;
            worker = WorkerFactory.CreateWorker(workerType, runtimeModel);
        }

        public void UpdateDiffuser(
            WorkerFactory.Type workerType, Model runtimeModel
        )
        {
            bool requiresUpdate = 
                workerType != this.WorkerType || runtimeModel != this.RuntimeModel;

            if(requiresUpdate)
            {
                Debug.Log("Diffuser update rewuitrd");
                Dispose();
                InitializeDiffuser(workerType, runtimeModel);
            }
        }

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
            Tensor initialNoise, 
            int modelOutputWidth,
            int modelOutputHeight,
            int diffusionSteps,
            int startingStep = 0, 
            int channels = 1,
            int batchSize = 1,
            float minSignalRate = 0.02f,
            float maxSignalRate = 0.9f
        )
        {
            float stepSize = 1.0f / diffusionSteps;

            Tensor nextNoisyImages = initialNoise;
            Tensor predictedImages = new Tensor(
                batchSize, modelOutputWidth, modelOutputHeight, channels
            );

            for(int step = startingStep; step < diffusionSteps; step++)
            {
                Tensor noisyImages = nextNoisyImages;

                float[] diffusionTimes = {1.0f - stepSize * step};
                Tensor[] rates = DiffusionSchedule(
                    diffusionTimes, minSignalRate, maxSignalRate
                );
                Tensor noiseRates = rates[0];
                Tensor signalRates = rates[1];
                Tensor noiseRatesSquared = tensorMathHelper.Pow(noiseRates, 2);
                
                worker.Execute(PackageInputs(noisyImages, noiseRatesSquared));
                Tensor predictedNoises = worker.PeekOutput();

                // Calculate predicted images.
                // predictedNoises * noiseRate
                Tensor scaledPredictedNoises = tensorMathHelper.ScaleBatches(
                    predictedNoises, noiseRates
                );

                // noisyImages - predictedNoises * noiseRate
                Tensor difference = tensorMathHelper.Sub(
                    noisyImages, scaledPredictedNoises
                );

                // (noisyImages - predictedNoises * noiseRate) / signalRate
                predictedImages = tensorMathHelper.ScaleBatches(
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

                Tensor[] nextRates = DiffusionSchedule(
                    nextDiffusionTimes, minSignalRate, maxSignalRate
                );
                Tensor nextNoiseRates = nextRates[0];
                Tensor nextSignalRates = nextRates[1];

                // predictedImages * nextSignalRate
                Tensor scaledPredictedImages = tensorMathHelper.ScaleBatches(
                    predictedImages, nextSignalRates
                );

                // predictedNoises * nextNoiseRate
                Tensor scaledPredictedNoises2 = tensorMathHelper.ScaleBatches(
                    predictedNoises, nextNoiseRates
                );

                // predictedImages * nextSignalRate + predictedNoises * nextNoiseRate
                nextNoisyImages = tensorMathHelper.Add(
                    scaledPredictedImages, scaledPredictedNoises2
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

        public Tensor Execute(
            Tensor input,
            int modelOutputWidth,
            int modelOutputHeight,
            int diffusionSteps, 
            int startingStep = 0
        )
        {
            Tensor output = ReverseDiffusion(
                input, modelOutputWidth, modelOutputHeight, diffusionSteps, startingStep
            );

            // TODO: I might have forgotten to denormalize values after reverse diffusion.
            // Reference this to make sure it was done correctly:
            // https://keras.io/examples/generative/ddim/

            return output;
        }

        public void Dispose()
        {
            if(IsDisposed)
            {
                Debug.LogError("Worker is already disposed");
                return;
            }

            worker.Dispose();
            IsDisposed = true;
        }
    }
}
