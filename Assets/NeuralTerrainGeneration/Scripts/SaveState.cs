using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralTerrainGeneration
{
    [CreateAssetMenu(
        fileName = "SaveState", 
        menuName = "ScriptableObjects/NTG/SaveState", 
        order = 1
    ), System.Serializable]
    public class SaveState : ScriptableObject
    {
        public Diffuser S_Diffuser;
        public BarraUpSampler S_BarraUpSampler;
        public GaussianSmoother S_GaussianSmoother;
    }
}