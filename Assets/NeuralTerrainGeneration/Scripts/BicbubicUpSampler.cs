using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    public class BicbubicUpSampler
    {
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

        public Tensor BicubicUpSample(Tensor original, int factor)
        {
            Tensor upSampledX = new Tensor(1, original.height, original.width * factor, 1);
            Tensor upSampled = new Tensor(
                1, original.height * factor, original.width * factor, 1
            );

            float p0 = 0;
            float p1 = 0;
            float p2 = 0;
            float p3 = 0;

            // Upsample width.
            for(int x = 0; x < original.width; x++)
            {
                for(int y = 0; y < original.height; y++)
                {
                    // p0
                    if(x - 1 < 0)
                    {
                        System.Random random = new System.Random();
                        p0 = (float)random.NextDouble();
                    }
                    else
                    {
                        p0 = original[0, y, x - 1, 0];
                    }

                    // p1
                    p1 = original[0, y, x, 0];

                    // p2 and p3
                    if(x + 2 >= original.width)
                    {
                        System.Random random = new System.Random();
                        p2 = (float)random.NextDouble();
                        p3 = (float)random.NextDouble();
                    }
                    else
                    {
                        p2 = original[0, y, x + 1, 0];
                        p3 = original[0, y, x + 2, 0];
                    }

                    float[] samples = SampleCubic(factor, p0, p1, p2, p3);
                    for(int i = 0; i < factor - 1; i++)
                    {
                        upSampledX[0, y, x * factor + i + 1, 0] = samples[i];
                    }
                    upSampledX [0, y, x * factor, 0] = p1;
                }
            }

            // Upsample height.
            for(int x = 0; x < upSampledX.width; x++)
            {
                for(int y = 0; y < original.height; y++)
                {
                    // p0
                    if(y - 1 < 0)
                    {
                        System.Random random = new System.Random();
                        p0 = (float)random.NextDouble();
                    }
                    else
                    {
                        p0 = upSampledX[0, y - 1, x, 0];
                    }

                    // p1
                    p1 = upSampledX[0, y, x, 0];

                    // p2 and p3
                    if(y + 2 >= original.height)
                    {
                        System.Random random = new System.Random();
                        p2 = (float)random.NextDouble();
                        p3 = (float)random.NextDouble();
                    }
                    else
                    {
                        p2 = upSampledX[0, y + 1, x, 0];
                        p3 = upSampledX[0, y + 2, x, 0];
                    }

                    float[] samples = SampleCubic(factor, p0, p1, p2, p3);
                    for(int i = 0; i < factor - 1; i++)
                    {
                        upSampled[0, y * factor + i + 1, x, 0] = samples[i];
                    }
                    upSampled[0, y * factor, x, 0] = p1;
                }
            }

            upSampledX.Dispose();
            return upSampled;
        }
    }
}
