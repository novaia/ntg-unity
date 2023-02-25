using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class GANTerrainGenerator : BaseTerrainGenerator
{
    [SerializeField] protected bool BigMountainTopLeft;
    [SerializeField] protected bool CentralValley;
    [SerializeField] protected bool Lowlands;
    [SerializeField] protected bool Highlands;
    [SerializeField] protected bool DiagonalRidge;
    [SerializeField] protected bool Highlands2;
    [SerializeField] protected bool CentralValley2;
    [SerializeField] protected bool BottomRightDecline;
    [SerializeField] protected bool BottomRightDecline2;
    [SerializeField] protected bool DivergingRidges;
    [SerializeField] protected bool Highlands3;
    [SerializeField] protected bool ValleyPass;
    [SerializeField] protected bool CentralValley3;
    [SerializeField] protected bool BottomLeftDecline;
    [SerializeField] protected bool randomNormal;

    protected LatentVectors latentVectors;

    public override void Setup()
    {
        base.Setup();
        latentVectors = new LatentVectors();
    }

    public float[] GenerateHeightmapFromLatent()
    {
        Tensor inputLatent = CustomInputTensor();

        object [] args = new object[] {inputLatent};
        float[] heightmap = GenerateHeightmap(runtimeModel, 
                                              new WorkerExecuter(GANStep),
                                              args);
        inputLatent.Dispose();
        return heightmap;
    }

    protected Tensor GANStep(IWorker worker, params object[] args)
    {
        Tensor input = (Tensor)args[0];
        worker.Execute(input);
        Tensor output = worker.PeekOutput();
        input.Dispose();
        return output;
    }

    // Ugly.
    protected Tensor CustomInputTensor()
    {
        Tensor input = new Tensor(1, 100);

        if(BigMountainTopLeft)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.BigMountainTopLeft)
            );
        }
        if(CentralValley)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.CentralValley)
            );
        }
        if(Lowlands)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.Lowlands)
            );
        }
        if(Highlands)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.Highlands)
            );
        }
        if(DiagonalRidge)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.DiagonalRidge)
            );
        }
        if(Highlands2)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.Highlands2)
            );
        }
        if(CentralValley2)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.CentralValley2)
            );
        }
        if(BottomRightDecline)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.BottomRightDecline)
            );
        }
        if(BottomRightDecline2)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.BottomRightDecline2)
            );
        }
        if(DivergingRidges)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.DivergingRidges)
            );
        }        
        if(DivergingRidges)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.DivergingRidges)
            );
        }
        if(ValleyPass)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.ValleyPass)
            );
        }
        if(CentralValley3)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.CentralValley3)
            );
        }
        if(BottomLeftDecline)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                InputTensorFromArray(latentVectors.BottomLeftDecline)
            );
        }
        if(randomNormal)
        {
            input = tensorMathHelper.AddTensor(
                input, 
                tensorMathHelper.RandomNormalTensor(1, 1, 100, 1)
            );
        }
        return input;
    }

    protected Tensor InputTensorFromArray(float[] inputArray)
    {
        Tensor input = new Tensor(1, 100);
        for(int i = 0; i < inputArray.Length; i++)
        {
            input[i] = inputArray[i];
        }
        return input;
    }
}
