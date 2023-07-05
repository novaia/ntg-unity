using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    public class TensorMathHelper
    {
        public Tensor NormalizeTensor(Tensor tensor)
        {
            float magnitude = 0.0f;
            for(int i = 0; i < tensor.length; i++)
            {
                magnitude += tensor[i] * tensor[i];
            }
            magnitude = Mathf.Sqrt(magnitude);

            Tensor normalizedTensor = new Tensor(
                tensor.batch, tensor.height, tensor.width, tensor.channels
            );
            for(int i = 0; i < tensor.length; i++)
            {
                normalizedTensor[i] = tensor[i] / magnitude;
            }
            return normalizedTensor;
        }

        // Treats tensors as single column vectors and performs dot product.
        public float VectorDotProduct(Tensor tensor1, Tensor tensor2)
        {
            if(tensor1.length != tensor2.length)
            {
                Debug.LogError("Tensors must be the same size.");
                return 0.0f;
            }

            float dotProduct = 0.0f;
            for(int i = 0; i < tensor1.length; i++)
            {
                dotProduct += tensor1[i] * tensor2[i];
            }
            return dotProduct;
        }

        // Treats tensors as single column vectors and slerps.
        public Tensor VectorSlerp(Tensor tensor1, Tensor tensor2, float interpValue)
        {
            Tensor normalizedTensor1 = NormalizeTensor(tensor1);
            Tensor normalizedTensor2 = NormalizeTensor(tensor2);
            
            float cosOmega = VectorDotProduct(tensor1, tensor2);
            // If the angle is greater than 90 degrees, negate 
            // one of the vectors to use the acute angle instead.
            if(cosOmega < 0.0f)
            {
                tensor2 = Scale(tensor2, -1.0f);
                cosOmega = -cosOmega;
            }

            // If the angle is very small, use linear interpolation 
            // instead to avoid numerical instability.
            if(cosOmega > 0.9999f)
            {
                Tensor lerpedTensor = Add(
                    tensor1, 
                    Scale(Sub(tensor2, tensor1), interpValue)
                );
                return lerpedTensor;
            }

            // Compute the sine of the angle between the vectors.
            float sinOmega = Mathf.Sqrt(1 - cosOmega * cosOmega);
            // Compute the angle between the vectors.
            float omega = Mathf.Acos(cosOmega); 
            // Compute the scale factor for the first vector,
            float tensor1Scale = Mathf.Sin((1 - interpValue) * omega) / sinOmega;
            // Compute the scale factor for the second vector.
            float tensor2Scale = Mathf.Sin(interpValue * omega) / sinOmega;
            // Compute the weighted sum of the two vectors.
            Tensor weightedSum = Add(
                Scale(tensor1, tensor1Scale), 
                Scale(tensor2, tensor2Scale)
            );

            return weightedSum;
        }

        public Tensor RandomNormalTensor(int batchSize, int width, int height, int channels)
        {
            Tensor tensor = new Tensor(batchSize, width, height, channels);
            System.Random random = new System.Random();
            for(int i = 0; i < width * height; i++)
            {
                // Box-Muller transform.
                // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
                double mean = 0.0f;
                double stdDev = 1.0f;
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double randomStdNormal = 
                    Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2.0f * Math.PI * u2);
                double randomNormal = mean + stdDev * randomStdNormal;
                tensor[i] = (float)randomNormal;
            }
            return tensor;
        }

        public Tensor PseudoRandomNormalTensor(
            int batchSize, int width, int height, int channels, int seed
        )
        {
            Tensor tensor = new Tensor(batchSize, width, height, channels);
            System.Random random = new System.Random(seed);
            for(int i = 0; i < width * height; i++)
            {
                // Box-Muller transform.
                // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
                double mean = 0.0f;
                double stdDev = 1.0f;
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double randomStdNormal = 
                    Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2.0f * Math.PI * u2);
                double randomNormal = mean + stdDev * randomStdNormal;
                tensor[i] = (float)randomNormal;
            }
            return tensor;
        }

        public Tensor ElementWiseLerp(Tensor tensor1, Tensor tensor2, float lerpValue)
        {
            if(tensor1.length != tensor2.length)
            {
                Debug.LogError("Tensors must be the same size.");
                return null;
            }

            Tensor newTensor = new Tensor(
                tensor1.batch, tensor1.height, tensor1.width, tensor1.channels
            );
            for(int i = 0; i < tensor1.length; i++)
            {
                newTensor[i] = Mathf.Lerp(tensor1[i], tensor2[i], lerpValue);
            }
            return newTensor;
        }

        public Tensor Add(Tensor tensor1, Tensor tensor2)
        {
            if(tensor1.length != tensor2.length)
            {
                Debug.LogError("Tensors must be the same size.");
                return null;
            }

            Tensor newTensor = new Tensor(
                tensor1.batch, tensor1.height, tensor1.width, tensor1.channels
            );
            for(int i = 0; i < tensor1.length; i++)
            {
                newTensor[i] = tensor1[i] + tensor2[i];
            }
            return newTensor;
        }

        public Tensor ScaleBatches(
            Tensor tensor, Tensor scalars, bool inverseScalars = false
        )
        {
            if(tensor.batch != scalars.length)
            {
                Debug.LogError("Tensor batch size must match the number of scalars.");
                return null;
            }

            Tensor newTensor = new Tensor(
                tensor.batch, tensor.width, tensor.height, tensor.channels
            );
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

        public Tensor Scale(Tensor tensor, float scalar)
        {
            Tensor newTensor = new Tensor(
                tensor.batch, tensor.width, tensor.height, tensor.channels
            );
            for(int i = 0; i < tensor.length; i++)
            {
                newTensor[i] = tensor[i] * scalar;
            }
            return newTensor;
        }

        public Tensor Sub(Tensor leftTensor, Tensor rightTensor)
        {
            if(leftTensor.length != rightTensor.length)
            {
                Debug.LogError("Tensors must be the same size.");
                return null;
            }

            Tensor newTensor = new Tensor(
                leftTensor.batch, leftTensor.width, leftTensor.height, leftTensor.channels
            );
            for(int batch = 0; batch < leftTensor.batch; batch++)
            {
                for(int x = 0; x < leftTensor.width; x++)
                {
                    for(int y = 0; y < leftTensor.height; y++)
                    {
                        newTensor[batch, y, x, 0] = 
                            leftTensor[batch, y, x, 0] - rightTensor[batch, y, x, 0];
                    }
                }
            }
            return newTensor;
        }

        public Tensor Pow(Tensor tensor, int power)
        {
            Tensor newTensor = new Tensor(
                tensor.batch, tensor.width, tensor.height, tensor.channels
            );
            for(int i = 0; i < tensor.length; i++)
            {
                newTensor[i] = Mathf.Pow(tensor[i], power);
            }
            return newTensor;
        }

        public Tensor Gradient(
            float leftValue, 
            float rightValue, 
            float topValue, 
            float bottomValue, 
            int width, 
            int height
        )
        {
            Tensor newTensor = new Tensor(1, width, height, 1);
            float lrGradient = (rightValue - leftValue) / (width - 1);
            float tbGradient = (topValue - bottomValue) / (height - 1);
            for(int x = 0; x < width; x++)
            {
                for(int y = 0; y < height; y++)
                {
                    float lrValue = leftValue + (lrGradient * x);
                    float tbValue = bottomValue + (tbGradient * y);
                    newTensor[0, y, x, 0] = lrValue + tbValue;
                }
            }
            return newTensor;
        }

        public Tensor Mul(Tensor left, Tensor right)
        {
            if(left.length != right.length)
            {
                Debug.LogError("Tensors must be the same size.");
                return null;
            }

            Tensor newTensor = new Tensor(left.batch, left.width, left.height, left.channels);
            for(int i = 0; i < left.length; i++)
            {
                newTensor[i] = left[i] * right[i];
            }
            return newTensor;
        }

        public Tensor TwoDimArrToTensor(float[,] array)
        {
            int width = array.GetLength(0);
            int height = array.GetLength(1);
            Tensor newTensor = new Tensor(1, width, height, 1);
            for(int x = 0; x < width; x++)
            {
                for(int y = 0; y < height; y++)
                {
                    newTensor[0, y, x, 0] = array[x, y];
                }
            }
            return newTensor;
        }

        public Tensor Mirror(Tensor tensor, bool mirrorX, bool mirrorY)
        {
            Tensor newTensor = new Tensor(
                tensor.batch, tensor.width, tensor.height, tensor.channels
            );

            int xSign = 1;
            int ySign = 1;
            int xStart = 0;
            int yStart = 0;

            if(mirrorX)
            {
                xSign = -1;
                xStart = tensor.width - 1;
            }
            if(mirrorY)
            {
                ySign = -1;
                yStart = tensor.height - 1;
            }

            for(int batch = 0; batch < tensor.batch; batch++)
            {
                for(int x = 0; x < tensor.width; x++)
                {
                    for(int y = 0; y < tensor.height; y++)
                    {
                        newTensor[batch, y, x, 0] = 
                            tensor[batch, yStart + ySign * y, xStart + xSign * x, 0];
                    }
                }
            }
            return newTensor;
        }

        public Tensor Concat(Tensor left, Tensor right)
        {
            bool shapesMatch = 
                left.batch == right.batch && 
                left.height == right.height && 
                left.channels == right.channels;

            if(!shapesMatch)
            {
                Debug.LogError("Tensors must have the same batch, height and channels.");
                return null;
            }

            Tensor newTensor = new Tensor(
                left.batch, left.width + right.width, left.height, left.channels
            );
            for(int batch = 0; batch < left.batch; batch++)
            {
                for(int x = 0; x < left.width; x++)
                {
                    for(int y = 0; y < left.height; y++)
                    {
                        for(int channel = 0; channel < left.channels; channel++)
                        {
                            newTensor[batch, y, x, channel] = left[batch, y, x, channel];
                        }
                    }
                }
                for(int x = 0; x < right.width; x++)
                {
                    for(int y = 0; y < right.height; y++)
                    {
                        for(int channel = 0; channel < right.channels; channel++)
                        {
                            newTensor[batch, y, x + left.width, channel] 
                                = right[batch, y, x, channel];
                        }
                    }
                }
            }
            return newTensor;
        }

        public Tensor Split(Tensor tensor)
        {
            if(tensor.width % 2 != 0)
            {
                Debug.LogError("Tensor width must be even.");
                return null;
            }

            Tensor newTensor = new Tensor(
                tensor.batch, tensor.width / 2, tensor.height, tensor.channels
            );
            for(int batch = 0; batch < tensor.batch; batch++)
            {
                for(int x = 0; x < newTensor.width; x++)
                {
                    for(int y = 0; y < newTensor.height; y++)
                    {
                        for(int channel = 0; channel < newTensor.channels; channel++)
                        {
                            newTensor[batch, y, x, channel] = tensor[batch, y, x, channel];
                        }
                    }
                }
            }
            return newTensor;
        }

        public Tensor Populated(float element, int width, int height)
        {
            Tensor populated = new Tensor(1, width, height, 1);
            for(int x = 0; x < width; x++)
            {
                for(int y = 0; y < height; y ++)
                {
                    populated[0, x, y, 0] = element;
                }
            }
            return populated;
        }
    }
}
