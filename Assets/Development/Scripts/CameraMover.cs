using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraMover : MonoBehaviour
{
    [SerializeField] private Transform target;
    [SerializeField] private float rotSpeed = 1f;
    [SerializeField] private float moveSpeed = 1f;
    [SerializeField] private Vector3 rotation;
    [SerializeField] private Vector3 moveDir;
    [SerializeField] private bool moveForward;
    [SerializeField] private bool localRotation = false;

    private void Start()
    {
        if(moveForward)
        {
            moveDir = Vector3.forward;
        }
    }

    private void LateUpdate()
    {
        if(localRotation)
        {
            transform.Rotate(rotation * rotSpeed * Time.deltaTime);
        }
        else
        {
            transform.RotateAround(target.position, rotation, rotSpeed * Time.deltaTime);
        }
        transform.Translate(moveDir * moveSpeed * Time.deltaTime);
    }
}
