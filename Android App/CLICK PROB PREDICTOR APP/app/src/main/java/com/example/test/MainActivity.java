package com.example.test;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class MainActivity extends AppCompatActivity implements View.OnClickListener{
    private static final String INPUT_NODE = "input"; // input tensor name
    private static final String OUTPUT_NODE = "output"; // output tensor name
    private static final String[] OUTPUT_NODES = {"output"};
    private static final int OUTPUT_SIZE = 2; // number of classes
    private static final int INPUT_SIZE = 55;
    private static final String TAG = "length length ";

    private AssetManager assetMngr;
    private TensorFlowInferenceInterface TFInference;

    android.content.res.Resources res ;

    Button[] buttons;
    TextView[] tv;
    double[][] input={{0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,10,67.33,15,2},
                      {0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,5,76.0,12,8},
                      {0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,10,60.0,19,4},
                      {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,11,87.0,3,5},
                      {0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,5,80.0,12,6},
                      {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,15,80.0,3,6},
                      {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,11,86.0,4,5}};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        assetMngr=this.getAssets();
        TFInference=new TensorFlowInferenceInterface(assetMngr,"model.pb");
        buttons = new Button[10];
        tv      = new TextView[10];
        res = getResources();
        for(int i=0; i<7; i++)
        {
            String buttonID = "b" + (i + 1);
            String tvID = "r" + (i + 1);
            int resID = res.getIdentifier(buttonID, "id", getPackageName());
            int res2ID = res.getIdentifier(tvID, "id", getPackageName());
            buttons[i] = ((Button) findViewById(resID));
            tv[i]      = (TextView) findViewById(res2ID);
            buttons[i].setOnClickListener(this);
        }
    }

    @Override
    public void onClick(View v) {

      int index = 0;
        for (int i = 0; i < buttons.length; i++)
        {
            if (buttons[i].getId() == v.getId())
            {
                index = i;
                break;
            }
        }
        predict(index);

    }

    private void predict(int i)
    {
        float[] input_ar=new float[INPUT_SIZE];
        for(int j=0;j<INPUT_SIZE;j++)
        {
            input_ar[j]=(float)input[i][j];
        }
        float[] output_ar = new float[2];

        // Log.d(TAG, "TF output: " + String.valueOf(INPUT_SIZE));

        TFInference.feed(INPUT_NODE, input_ar,1,INPUT_SIZE);
        TFInference.run(OUTPUT_NODES,false);
        TFInference.fetch(OUTPUT_NODE,output_ar);
        tv[i].setText(String.valueOf(100*output_ar[1]) + "%");
    }
    public void resetAll(View v)
    {
        for(int i=0;i<7;i++)
        {
            tv[i].setText("press predict");
        }
    }
}
