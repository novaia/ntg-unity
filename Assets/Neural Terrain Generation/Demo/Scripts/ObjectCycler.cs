using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralTerrainGeneration.Demo
{
    public class ObjectCycler : MonoBehaviour
    {
        [SerializeField] private GameObject[] objects;
        [SerializeField] private float cycleTime = 5.0f;
        [SerializeField] private float currentCycleTime = 0.0f;
        private int currentObjectIndex = 0;
        [SerializeField] private bool repeatCycle = true;

        private void Start()
        {
            currentCycleTime = cycleTime;
            for(int i = 0; i < objects.Length; i++)
            {
                objects[i].SetActive(false);
            }
            if(objects.Length > 0)
            {
                objects[0].SetActive(true);
            }
        }

        private void Update()
        {
            currentCycleTime -= Time.deltaTime;
            if(currentCycleTime <= 0)
            {
                currentCycleTime = cycleTime;
                objects[currentObjectIndex].SetActive(false);
                currentObjectIndex++;
                if(repeatCycle)
                {
                    currentObjectIndex = (currentObjectIndex >= objects.Length) ? 0 : currentObjectIndex;
                }
                else
                {
                    currentObjectIndex = (currentObjectIndex >= objects.Length) ? currentObjectIndex-1 : currentObjectIndex;
                }
                objects[currentObjectIndex].SetActive(true);
            }
        }
    }
}