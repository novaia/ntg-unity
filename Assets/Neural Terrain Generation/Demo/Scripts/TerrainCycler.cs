using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralTerrainGeneration.Demo
{
    public class TerrainCycler : MonoBehaviour
    {
        [SerializeField] private Terrain[] terrains;
        [SerializeField] private float cycleTime = 5.0f;
        [SerializeField] private float currentCycleTime = 0.0f;
        private int currentTerrainIndex = 0;

        private void Start()
        {
            currentCycleTime = cycleTime;
        }

        private void Update()
        {
            currentCycleTime -= Time.deltaTime;
            if(currentCycleTime <= 0)
            {
                currentCycleTime = cycleTime;
                terrains[currentTerrainIndex].gameObject.SetActive(false);
                currentTerrainIndex++;
                currentTerrainIndex = (currentTerrainIndex >= terrains.Length) ? 0 : currentTerrainIndex;
                terrains[currentTerrainIndex].gameObject.SetActive(true);
            }
        }
    }
}