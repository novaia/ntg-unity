using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(Benchmarking))]
public class BenchmarkingEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        Benchmarking benchmarking = (Benchmarking)target;

        if(GUILayout.Button("BARRA Add Benchmark"))
        {
            benchmarking.BarraAddBenchmark();
        }
        if(GUILayout.Button("Normal Add Benchmark"))
        {
            benchmarking.NormalAddBenchmark();
        }
    }
}