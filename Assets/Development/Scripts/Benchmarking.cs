using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class Benchmarking : MonoBehaviour
{
    [SerializeField] private int numTrials = 100;
    
    public void NormalAdd(Tensor tensor1, Tensor tensor2)
    {
        Tensor newTensor = new Tensor(tensor1.batch, tensor1.height, tensor1.width, tensor1.channels);
        for(int i = 0; i < tensor1.length; i++)
        {
            newTensor[i] = tensor1[i] + tensor2[i];
        }

        tensor1.Dispose();
        tensor2.Dispose();
        newTensor.Dispose();
    }

    private void BarraAdd(Tensor tensor1, Tensor tensor2)
    {
        ModelBuilder builder = new ModelBuilder();
        object[] inputs = new object[] 
        { 
            builder.Const("tensor1", tensor1).name,
            builder.Const("tensor2", tensor2).name 
        };
        Layer addLayer = builder.Add("Add", inputs);
        builder.Output(addLayer);
        Model model = builder.model;

        IWorker worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        worker.Execute();
        Tensor output = worker.PeekOutput();

        tensor1.Dispose();
        tensor2.Dispose();
        output.Dispose();
        worker.Dispose();
    }

    public void BarraAddBenchmark()
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        for(int i = 0; i < numTrials; i++)
        {
            BarraAdd(
                PopulatedTensor(2, 100, 100),
                PopulatedTensor(1, 100, 100)
            );
        }
        watch.Stop();
        Debug.Log("BarraAdd: " + watch.ElapsedMilliseconds + "ms");
    }

    public void NormalAddBenchmark()
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        for(int i = 0; i < numTrials; i++)
        {
            NormalAdd(
                PopulatedTensor(2, 100, 100),
                PopulatedTensor(1, 100, 100)
            );
        }
        watch.Stop();
        Debug.Log("NormalAdd: " + watch.ElapsedMilliseconds + "ms");
    }

    public Tensor PopulatedTensor(float element, int width, int height)
    {
        Tensor populatedTensor = new Tensor(1, width, height, 1);
        for(int x = 0; x < width; x++)
        {
            for(int y = 0; y < height; y ++)
            {
                populatedTensor[0, x, y, 0] = element;
            }
        }
        return populatedTensor;
    }
}
