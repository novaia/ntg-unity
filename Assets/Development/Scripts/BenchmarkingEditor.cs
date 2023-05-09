using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(Benchmarking))]
public class BenchmarkingEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        Benchmarking benchmarking = (Benchmarking)target;

        if(GUILayout.Button("Barra Add Benchmark"))
        {
            benchmarking.BarraAddBenchmark();
        }
        if(GUILayout.Button("Normal Add Benchmark"))
        {
            benchmarking.NormalAddBenchmark();
        }
        if(GUILayout.Button("Barra Mul Benchmark"))
        {
            benchmarking.BarraMulBenchmark();
        }
        if(GUILayout.Button("Barra Mul Prebuilt Benchmark"))
        {
            benchmarking.BarraMulPrebuiltBenchmark();
        }
        if(GUILayout.Button("Normal Mul Benchmark"))
        {
            benchmarking.NormalMulBenchmark();
        }
        if(GUILayout.Button("Burst Mul Benchmark"))
        {
            benchmarking.BurstMulBenchmark();
        }
        if(GUILayout.Button("Unsafe Mul Benchmark"))
        {
            benchmarking.UnsafeMulBenchmark();
        }
        if(GUILayout.Button("Barra Upsample Benchmark"))
        {
            benchmarking.BarraUpsampleBenchmark();
        }
        if(GUILayout.Button("Normal Upsample Benchmark"))
        {
            benchmarking.NormalUpsampleBenchmark();
        }
    }
}