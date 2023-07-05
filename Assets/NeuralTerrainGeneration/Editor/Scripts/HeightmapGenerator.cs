using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    public class HeightmapGenerator
    {
        private TensorMathHelper tensorMathHelper = new TensorMathHelper();
        private const int batchSize = 1;
        private const int channels = 1;

        public float[] GenerateHeightmapFromScratch(
            int modelOutputWidth, 
            int modelOutputHeight, 
            int samplingSteps, 
            int seed, 
            bool smooth, 
            BarraUpSampler barraUpSampler, 
            GaussianSmoother gaussianSmoother, 
            Diffuser diffuser
        )
        {
            Tensor baseHeightmap = GenerateBaseHeightmap(
                modelOutputWidth, modelOutputHeight, samplingSteps, seed, diffuser
            );

            Tensor upSampled = barraUpSampler.Execute(baseHeightmap);

            float[] finalHeightmap;
            if(smooth)
            {
                Tensor smoothed = gaussianSmoother.Execute(upSampled);
                finalHeightmap = smoothed.ToReadOnlyArray();
                smoothed.Dispose();
            }
            else
            {
                finalHeightmap = upSampled.ToReadOnlyArray();
            }

            baseHeightmap.Dispose();
            upSampled.Dispose();
            return finalHeightmap;
        }

        public Texture2D GenerateBrushHeightmap(
            int modelOutputWidth,
            int modelOutputHeight,
            int samplingSteps,
            int seed,
            bool smooth,
            float heightOffset,
            Texture2D brushMask,
            BarraUpSampler barraUpSampler,
            GaussianSmoother gaussianSmoother,
            Diffuser diffuser
        )
        {
            Tensor baseHeightmap = GenerateBaseHeightmap(
                modelOutputWidth, modelOutputHeight, samplingSteps, seed, diffuser
            );

            Tensor brushMaskTensor = new Tensor(brushMask, 1);
            Tensor maskedHeightmap = new Tensor(
                batchSize, modelOutputWidth, modelOutputHeight, channels
            );

            // Mask the base heightmap with given brush mask.
            for(int i = 0; i < maskedHeightmap.length; i++)
            {
                maskedHeightmap[i] = baseHeightmap[i] * brushMaskTensor[i];
            }

            // UpSample.
            Tensor upSampled = barraUpSampler.Execute(maskedHeightmap);

            // Smooth.
            Tensor finalHeightmap;
            if(smooth)
            {
                finalHeightmap = gaussianSmoother.Execute(upSampled);
            }
            else
            {
                finalHeightmap = upSampled;
            }

            // Convert the final heightmap to a texture.
            Texture2D finalHeightmapTexture = new Texture2D(
                finalHeightmap.width, finalHeightmap.height
            );

            Color[] finalHeightmapColors = new Color[finalHeightmap.length];
            for(int i = 0; i < finalHeightmap.length; i++)
            {
                // Multiplying by 4 is a hack to prevent the brush pixels
                // from becoming too compressed and losing information
                // which leads to staircasing.
                float colorValue = finalHeightmap[i] * 4.0f - heightOffset;
                finalHeightmapColors[i] = new Color(colorValue, colorValue, colorValue);
            }

            finalHeightmapTexture.SetPixels(
                0, 0, finalHeightmap.width, finalHeightmap.height, finalHeightmapColors
            );
            finalHeightmapTexture.Apply();

            baseHeightmap.Dispose();
            brushMaskTensor.Dispose();
            maskedHeightmap.Dispose();
            upSampled.Dispose();
            finalHeightmap.Dispose();
            return finalHeightmapTexture;
        }

        private Tensor GenerateBaseHeightmap(
            int modelOutputWidth,
            int modelOutputHeight,
            int samplingSteps, 
            int seed,
            Diffuser diffuser
        )
        {
            Tensor input = tensorMathHelper.PseudoRandomNormalTensor(
                batchSize, modelOutputWidth, modelOutputHeight, channels, seed
            );

            Tensor baseHeightmap = diffuser.Execute(
                input, modelOutputWidth, modelOutputHeight, samplingSteps
            );

            input.Dispose();
            return baseHeightmap;
        }
    }
}
