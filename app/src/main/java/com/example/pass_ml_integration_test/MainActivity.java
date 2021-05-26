package com.example.pass_ml_integration_test;


import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import com.example.pass_ml_integration_test.ml.Model;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    private Button predict;
    private TextView one, two , result;
    private float[] numbers = {1, 2, 3, 2, 3, 1, 1, 2, 3, 2, 3, 1, 1, 2, 3, 2, 3, 1, 2, 1, 2};
    private String output;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        predict = (Button) findViewById(R.id.button2);
        one = (TextView) findViewById(R.id.one);
        two = (TextView) findViewById(R.id.two);
        result = (TextView) findViewById(R.id.result);

        predict.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                try {
                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 21}, DataType.FLOAT32);

                  //  ByteBuffer.allocate(4).putFloat(numbers[0]).array();
                    byte[] byteArray= FloatArray2ByteArray(numbers);
                    ByteBuffer byteBuffer= ByteBuffer.wrap(byteArray);

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    String converted= new String(byteBuffer.array(), "UTF-8");


                    if (outputFeature0.getFloatArray()[0] > outputFeature0.getFloatArray()[1]){
                        output = "Autistic";
                    }
                    else if(outputFeature0.getFloatArray()[1] > outputFeature0.getFloatArray()[0]){
                        output = "Not autistic";
                  }

                    one.setText("" + outputFeature0.getFloatArray()[0]);
                    two.setText("" + outputFeature0.getFloatArray()[1]) ;
                    result.setText(output);
                    Toast.makeText(MainActivity.this, "output:" + outputFeature0.getFloatArray()[0] + " , " + outputFeature0.getFloatArray()[1], Toast.LENGTH_SHORT).show();

                    // Releases model resources if no longer used.
                    model.close();




                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });
    }
    public static byte[] FloatArray2ByteArray(float[] values){
        ByteBuffer buffer= ByteBuffer.allocate(4 * values.length);
        for (float value : values){
            buffer.putFloat(value);
        }
        return buffer.array();
    }

}