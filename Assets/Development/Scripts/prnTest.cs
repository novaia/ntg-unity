using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class prnTest : MonoBehaviour
{
    [SerializeField] private int seed;
    [SerializeField] private bool randomSeed;

    private void Start()
    {
        if(randomSeed)
        {
            seed = UnityEngine.Random.Range(0, 100000);
        }
        // Create a random number generator with a seed
        System.Random random = new System.Random(seed);

        // Create a list of 10 numbers
        List<int> numbers = new List<int>();
        for (int i = 0; i < 10; i++)
        {
            // Generate a random number between 0 and 100
            int number = random.Next(0, 101);
            // Add it to the list
            numbers.Add(number);
        }

        foreach (int number in numbers)
        {
            Debug.Log(number);
        }
    }
}