Code releases for PhotoReg.

Our website: https://ziweny11.github.io/photoreg/

Our paper on Arxiv:

Setup:
1. Clone the 3DGS repo from their website: https://github.com/graphdeco-inria/gaussian-splatting
     ```
    git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
    ```
2. Clone the DUSt3R repo from their website: https://github.com/naver/dust3r
   ```
    git clone --recursive https://github.com/naver/dust3r
    ```
3. Clone our repo:
    ```
    git clone --recursive https://github.com/ziweny11/PhotoRegCodes
    ```
4. run our code on two or more generated 3DGS model:
    ```
    python main.py --checkpoint_multi <path to trained model1> <path to trained model2>
    ```
5. Visualization:
    ```
    python gs_viz.py --checkpoint gnew.pth
    ```
6. GS cloud Visualization:
    ```
    python PC_viz.py --checkpoint gnew.pth
    ```
