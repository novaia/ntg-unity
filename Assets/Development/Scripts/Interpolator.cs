using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;
using System;

// This is not for use in the final product. 
// It is just a script used to showoff interpolation.
public class Interpolator : MonoBehaviour
{
    // General.
    [SerializeField] private WorkerFactory.Type workerType = WorkerFactory.Type.ComputePrecompiled;
    [SerializeField] private TensorMathHelper tensorMathHelper = new TensorMathHelper();
    [SerializeField] private TerrainHelper terrainHelper = new TerrainHelper();
    [SerializeField] private int modelOutputWidth = 256;
    [SerializeField] private int modelOutputHeight = 256;
    private int upSampledWidth = 0;
    private int upSampledHeight = 0;
    [SerializeField] private float heightMultiplier = 0.5f;
    private int channels = 1;
    [SerializeField] private NNModel modelAsset;
    private Model runtimeModel;

    // Diffusion.
    private const float maxSignalRate = 0.9f;
    private const float minSignalRate = 0.02f;
    private Diffuser diffuser = new Diffuser();
    private int samplingSteps = 10;

    // Upsampling.
    // Left: upsample resolution, right: upsample factor.
    private enum UpSampleResolution 
    {
        _256 = 1, 
        _512 = 2, 
        _1024 = 4, 
        _2048 = 8, 
        _4096 = 16
    };
    private UpSampleResolution upSampleResolution = UpSampleResolution._512;

    // Smoothing.
    [SerializeField] private int kernelSize = 12;
    [SerializeField] private float sigma = 6.0f;

    // Seeds to interpolate.
    [SerializeField] private int[] seeds = new int[2];

    // Interpolation.
    private const int numInterpolationSteps = 20;
    private float interpolationStepSize = 0.0f;
    private float[] stepSizes = new float[numInterpolationSteps];

    [SerializeField] private Terrain terrain;

    private void Start()
    {
        // Calculate upsampled dimensions.
        int upSampleFactor = (int)upSampleResolution;
        upSampledWidth = modelOutputWidth * upSampleFactor;
        upSampledHeight = modelOutputHeight * upSampleFactor;

        // Load model.
        runtimeModel = ModelLoader.Load(modelAsset);

        // Calculate interpolation step size.
        interpolationStepSize = 1.0f / (numInterpolationSteps);

        for(int i = 0; i < numInterpolationSteps; i++)
        {
            stepSizes[i] = StepSizeSchedule(i, numInterpolationSteps);
        }
        float stepSizesSum = 0.0f;
        for(int i = 0; i < numInterpolationSteps; i++)
        {
            stepSizesSum += stepSizes[i];
        }
        for(int i = 0; i < numInterpolationSteps; i++)
        {
            stepSizes[i] /= stepSizesSum;
        }

        StartCoroutine(MyCoroutine());
    }

    private float StepSizeSchedule(float currentStep, float endPoint)
    {
        /*double n = (float)endPoint;
        double k = (float)steepness;
        double x = (float)currentStep;
        double y = 1 / (1 + Math.Exp(-k * (x - n / 2)));
        double dy = k * y * (1 - y);
        return (float)y;*/

        return -0.25f * currentStep * (currentStep - endPoint);
    }

    private IEnumerator MyCoroutine()
    {
        for(int currentSeedIndex = 0; currentSeedIndex < seeds.Length - 1; currentSeedIndex++)
        {
            int seed1 = seeds[currentSeedIndex];
            int seed2 = seeds[currentSeedIndex + 1];

            Tensor input1 = tensorMathHelper.PseudoRandomNormalTensor(
                1,
                modelOutputWidth,
                modelOutputHeight,
                channels,
                seed1
            );
            Tensor input2 = tensorMathHelper.PseudoRandomNormalTensor(
                1,
                modelOutputWidth,
                modelOutputHeight,
                channels,
                seed2
            );

            for(int currentStep = 0; currentStep < numInterpolationSteps; currentStep++)
            {
                float t = 0.0f;
                for(int i = 0; i < currentStep; i++)
                {
                    t += stepSizes[i];
                }
                Tensor interpolatedInput = tensorMathHelper.VectorSlerp(
                    input1,
                    input2,
                    t
                );
                float[] heightmap = GenerateHeightmap(
                    upSampleResolution,
                    samplingSteps,
                    0,
                    interpolatedInput
                );
                terrainHelper.SetTerrainHeights(
                    terrain,
                    heightmap,
                    upSampledWidth,
                    upSampledHeight,
                    heightMultiplier
                );
                yield return new WaitForSeconds(2);
            }
            Debug.Log("Done with seed " + currentSeedIndex);
        }
    }

    private Tensor Smooth(Tensor input)
    {
        GaussianSmoother gaussianSmoother = new GaussianSmoother(
            kernelSize, 
            sigma,
            1,
            kernelSize-1, 
            upSampledWidth, 
            upSampledHeight,
            workerType
        );
        Tensor output = gaussianSmoother.Execute(input);
        gaussianSmoother.Dispose();

        return output;
    }

    private Tensor UpSample(Tensor input, UpSampleResolution upSampleResolutionArg)
    {
        if(upSampleResolutionArg == UpSampleResolution._256)
        {
            return input;
        }

        int upSampleFactor = (int)upSampleResolutionArg;
        Tensor output = new Tensor(1, upSampledHeight, upSampledWidth, 1);

        BarraUpSampler barraUpSampler = new BarraUpSampler(
            modelOutputWidth,
            modelOutputHeight,
            upSampleFactor, 
            true,
            workerType
        );
        output = barraUpSampler.Execute(input);
        barraUpSampler.Dispose();
        
        return output;
    }

    private float[] GenerateHeightmap(
        UpSampleResolution upSampleResolutionArg, 
        int diffusionSteps, 
        int startingStep = 0, 
        Tensor customInput = null
    )
    {
        float[] heightmap = new float[upSampledWidth * upSampledHeight];
        using(var worker = WorkerFactory.CreateWorker(workerType, runtimeModel))
        {
            Tensor input = new Tensor(1, modelOutputWidth, modelOutputHeight, channels);
            input = customInput;

            Tensor diffusionOutput = diffuser.ReverseDiffusion(
                worker, 
                input, 
                modelOutputWidth, 
                modelOutputHeight,
                diffusionSteps,
                startingStep
            );

            Tensor upSampled = UpSample(diffusionOutput, upSampleResolutionArg);
            Tensor smoothed = Smooth(upSampled);
            heightmap = smoothed.ToReadOnlyArray();
            input.Dispose();
            diffusionOutput.Dispose();
            upSampled.Dispose();
            smoothed.Dispose();
        }

        // TODO: I might have forgotten to denormalize values after reverse diffusion.
        // Reference this to make sure it was done correctly:
        // https://keras.io/examples/generative/ddim/

        return heightmap;
    }
}
