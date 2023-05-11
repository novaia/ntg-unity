using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;
using System;

namespace NeuralTerrainGeneration
{
    public class GaussianSmoother
    {
        public int KernelSize { get; private set; }
        public float Sigma { get; private set; }
        public float Stride { get; private set; }
        public float Pad { get; private set; }
        public int InputWidth { get; private set; }
        public int InputHeight { get; private set; }
        public bool IsDisposed { get; private set; }
        public WorkerFactory.Type WorkerType { get; private set; }
        private IWorker worker;
        private Tensor kernel;
        private const string inputName = "input";
        private TensorMathHelper tensorMathHelper = new TensorMathHelper();

        public GaussianSmoother(
            int kernelSize, 
            float sigma,
            int stride,
            int pad, 
            int inputWidth, 
            int inputHeight
        )
        {
            this.KernelSize = kernelSize;
            this.Sigma = sigma;
            this.Stride = stride;
            this.Pad = pad;

            ModelBuilder builder = new ModelBuilder();
            builder.Input(inputName, 1, inputHeight, inputWidth, 1);
            object input = inputName;

            kernel = CreateKernel(kernelSize, sigma);
            //Tensor bias = tensorMathHelper.PopulatedTensor(0, kernelSize, kernelSize);
            Tensor bias = new Tensor(1, 1, 1, 1);
            bias[0] = 0;
            Int32[] strideArray = new Int32[] { stride, stride };
            //Int32[] padArray = new Int32[] { inputWidth, inputHeight, inputWidth, inputHeight };
            Int32[] padArray = new Int32[] { pad, pad, 0, 0 };

            Layer convLayer = builder.Conv2D(
                "Conv2D", 
                input, 
                strideArray, 
                padArray, 
                kernel, 
                bias
            );

            builder.Output(convLayer);
            Model model = builder.model;

            worker = WorkerFactory.CreateWorker(WorkerType, model);
        }

        public Tensor CreateKernel(int size, float sigma)
        {
            Tensor kernelTensor = new Tensor(size, size, 1, 1);
            for(int i = 0; i < kernelTensor.length; i++)
            {
                int x = i % size;
                int y = i / size;
                float value = (float)(
                    1 / (2 * Math.PI * sigma * sigma) * 
                    Math.Exp(-((x * x + y * y) / (2 * sigma * sigma)))
                );
                kernelTensor[i] = value;
            }
            return kernelTensor;
        }

        public Tensor Execute(Tensor inputTensor)
        {
            if(IsDisposed)
            {
                Debug.LogError("Worker is disposed");
                return null;
            }

            worker.Execute(inputTensor);
            return worker.PeekOutput();
        }

        public void Dispose()
        {
            if(IsDisposed)
            {
                Debug.LogError("Worker is already disposed");
                return;
            }

            kernel.Dispose();
            worker.Dispose();
            IsDisposed = true;
        }
    }
}
