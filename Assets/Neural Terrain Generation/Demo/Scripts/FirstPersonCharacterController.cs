using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralTerrainGeneration.Demo
{
    public class FirstPersonCharacterController : MonoBehaviour 
    {
        [SerializeField] private float moveSpeed;
        [SerializeField] private float gravityForce;
        [SerializeField] private float sensitivityX;
        [SerializeField] private float sensitivityY;
        [SerializeField] private Camera mainCamera;
        private CharacterController Controller;
        private Quaternion cameraTargetRot;

        private void Start()
        {
            Controller = GetComponent<CharacterController>();
            cameraTargetRot = mainCamera.transform.localRotation;
        }

        private void Update()
        {
            // Rotation.
            float mouseX = Input.GetAxis("Mouse X");
            mouseX *= sensitivityX * Time.deltaTime;
            transform.Rotate(Vector3.up, mouseX);

            float mouseY = Input.GetAxis("Mouse Y");
            mouseY *= sensitivityY * Time.deltaTime;
            cameraTargetRot *= Quaternion.Euler(-mouseY, 0, 0);
            cameraTargetRot = ClampRotationAroundXAxis(cameraTargetRot);
            mainCamera.transform.localRotation = cameraTargetRot;

            // Movement.
            float inputX = Input.GetAxisRaw("Horizontal");
            float inputY = Input.GetAxisRaw("Vertical");

            Vector3 movement = new Vector3(inputX, 0, inputY);
            movement.Normalize();
            movement = transform.TransformDirection(movement);
            movement *= moveSpeed;
            movement = new Vector3(movement.x, gravityForce, movement.z);
            movement *= Time.deltaTime;

            Controller.Move(movement);
        }

        private Quaternion ClampRotationAroundXAxis(Quaternion q)
        {
            q.x /= q.w;
            q.y /= q.w;
            q.z /= q.w;
            q.w = 1.0f;

            float angleX = 2.0f * Mathf.Rad2Deg * Mathf.Atan (q.x);

            angleX = Mathf.Clamp (angleX, -90, 90);

            q.x = Mathf.Tan (0.5f * Mathf.Deg2Rad * angleX);

            return q;
        }
    }
}