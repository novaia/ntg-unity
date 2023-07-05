using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

namespace NeuralTerrainGeneration
{
    [System.Serializable]
    public class BarraUpSampler
    {
        public int Factor { get; private set; }
        public bool Bilinear { get; private set; }
        public int InputWidth { get; private set; }
        public int InputHeight { get; private set; }
        public bool IsDisposed { get; private set; }
        public WorkerFactory.Type WorkerType { get; private set; }
        private IWorker worker;

        private const string inputName = "input";

        public BarraUpSampler(
            int inputWidth, 
            int inputHeight, 
            int factor, 
            bool bilinear, 
            WorkerFactory.Type workerType
        )
        {
            InitializeUpSampler(inputWidth, inputHeight, factor, bilinear, workerType);
        }

        private void InitializeUpSampler(
            int inputWidth, 
            int inputHeight, 
            int factor, 
            bool bilinear, 
            WorkerFactory.Type workerType 
        )
        {
            this.IsDisposed = false;
            this.Factor = factor;
            this.Bilinear = bilinear;
            this.InputWidth = inputWidth;
            this.InputHeight = inputHeight;
            this.WorkerType = workerType;

            ModelBuilder builder = new ModelBuilder();
            
            builder.Input(inputName, 1, this.InputHeight, this.InputWidth, 1);
            object input = inputName;

            Int32[] upSampleFactor = new Int32[] { this.Factor, this.Factor };
            Layer upSampleLayer = builder.Upsample2D(
                "Upsample2D", 
                input, 
                upSampleFactor, 
                this.Bilinear
            );
            
            builder.Output(upSampleLayer);
            Model model = builder.model;

            worker = WorkerFactory.CreateWorker(WorkerType, model);
        }

        public void UpdateUpSampler(
            int inputWidth, 
            int inputHeight, 
            int factor, 
            bool bilinear, 
            WorkerFactory.Type workerType 
        )
        {
            bool requiresUpdate = 
                inputWidth != this.InputWidth || 
                inputHeight != this.InputHeight || 
                factor != this.Factor || 
                bilinear != this.Bilinear || 
                workerType != this.WorkerType;

            if(requiresUpdate)
            {
                Dispose();
                InitializeUpSampler(inputWidth, inputHeight, factor, bilinear, workerType);
            }
        }

        public Tensor Execute(Tensor inputTensor)
        {
            if(IsDisposed)
            {
                Debug.LogError("Worker is disposed");
                return null;
            }

            IDictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
            inputs.Add(inputName, inputTensor);
            worker.Execute(inputs);
            Tensor output = worker.PeekOutput();
            output.TakeOwnership(); // Take ownership so tensor can outlive worker.
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
