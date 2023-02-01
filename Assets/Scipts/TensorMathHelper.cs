using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class TensorMathHelper
{
    public Tensor RandomNormalTensor(int batchSize, int width, int height, int channels)
    {
        Tensor tensor = new Tensor(batchSize, width, height, channels);
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

    public Tensor AddTensor(Tensor tensor1, Tensor tensor2)
    {
        if(tensor1.length != tensor2.length)
        {
            Debug.LogError("Tensors must be the same size.");
            return null;
        }

        Tensor newTensor = new Tensor(tensor1.batch, tensor1.height, tensor1.width, tensor1.channels);
        for(int i = 0; i < tensor1.length; i++)
        {
            newTensor[i] = tensor1[i] + tensor2[i];
        }
        return newTensor;
    }

    public Tensor ScaleTensorBatches(Tensor tensor, Tensor scalars, bool inverseScalars = false)
    {
        if(tensor.batch != scalars.length)
        {
            Debug.LogError("Tensor batch size must match the number of scalars.");
            return null;
        }

        Tensor newTensor = new Tensor(tensor.batch, tensor.width, tensor.height, tensor.channels);
        for(int batch = 0; batch < tensor.batch; batch++)
        {
            float scalar = scalars[batch];
            if(inverseScalars)
            {
                scalar = 1.0f / scalar;
            }
            for(int x = 0; x < tensor.width; x++)
            {
                for(int y = 0; y < tensor.height; y++)
                {
                    newTensor[batch, y, x, 0] = tensor[batch, y, x, 0] * scalar;
                }
            }
        }
        return newTensor;
    }

    public Tensor ScaleTensor(Tensor tensor, float scalar)
    {
        Tensor newTensor = new Tensor(tensor.batch, tensor.width, tensor.height, tensor.channels);
        for(int i = 0; i < tensor.length; i++)
        {
            newTensor[i] = tensor[i] * scalar;
        }
        return newTensor;
    }

    public Tensor SubtractTensor(Tensor leftTensor, Tensor rightTensor)
    {
        if(leftTensor.length != rightTensor.length)
        {
            Debug.LogError("Tensors must be the same size.");
            return null;
        }

        Tensor newTensor = new Tensor(leftTensor.batch, leftTensor.width, leftTensor.height, leftTensor.channels);
        for(int batch = 0; batch < leftTensor.batch; batch++)
        {
            for(int x = 0; x < leftTensor.width; x++)
            {
                for(int y = 0; y < leftTensor.height; y++)
                {
                    newTensor[batch, y, x, 0] = leftTensor[batch, y, x, 0] - rightTensor[batch, y, x, 0];
                }
            }
        }
        /*for(int i = 0; i < leftTensor.length; i++)
        {
            newTensor[i] = leftTensor[i] - rightTensor[i];
        }*/
        return newTensor;
    }

    public Tensor RaiseTensorToPower(Tensor tensor, int power)
    {
        Tensor newTensor = new Tensor(tensor.batch, tensor.width, tensor.height, tensor.channels);
        for(int i = 0; i < tensor.length; i++)
        {
            newTensor[i] = Mathf.Pow(tensor[i], power);
        }
        return newTensor;
    }
}
