package com.example.suji.define;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;

import java.io.ByteArrayOutputStream;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

@SuppressWarnings("deprecation")
public class MainActivity extends Activity implements Camera.PictureCallback, Camera.ShutterCallback, SurfaceHolder.Callback {

    private TextView textcaptured;
    private Button take_photo, capture;
    private  FocusBoxView focusBox;
    private  CameraView camera_views;
    private  byte[] byteArray;
    private String base64array;
    Bitmap sendbmp;
    boolean jpgbitmap;
    JSONArray mJSONArray;
    private String  url = " http:/192.168.0.20:5000";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textcaptured = (TextView)findViewById(R.id.textView);
        capture = (Button)findViewById(R.id.capture);
        focusBox= (FocusBoxView)findViewById(R.id.focus_box);
        Log.d("Message:", "Camera opened and inside if");
        SurfaceView camera_view = (SurfaceView)findViewById(R.id.camera_view);
        SurfaceHolder surfaceHolder = camera_view.getHolder();
        camera_views = new CameraView(surfaceHolder) ;
        surfaceHolder.addCallback(camera_views);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.

        return true;
    }

    protected void onResume() {
        super.onResume();


    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement


        return true;
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){

    }

    public void Capture_image(View view){
        try{
            CameraView.takeShot(this, this, this);
        }catch (Exception e){
            Log.d("Error", "Inside Camera_image. Camera cannot take picture " +e.getMessage());
        }
    }

    @Override
    public void onShutter() {

    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {

    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {


        Log.d("Message:", "Camera opened and inside if");
        SurfaceView camera_view = (SurfaceView)findViewById(R.id.camera_view);
        SurfaceHolder surfaceHolder = camera_view.getHolder();
        camera_views = new CameraView(surfaceHolder) ;
        surfaceHolder.addCallback(camera_views);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);


    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {

    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {
        Log.d("Message", "Picture taken");

        if (data == null) {
            Log.d("Message", "Got null data");
            return;
        }
        sendbmp = Tools.getFocusedBitmap(getApplicationContext(), camera, data, focusBox.getBox());

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        jpgbitmap = sendbmp.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        byteArray = stream.toByteArray();
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                mJSONArray = new JSONArray(byteArray);
                base64array = Base64.encodeToString(byteArray,Base64.DEFAULT);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }

       new Networkconnection().execute();


        //Log.d("Step:", String.valueOf(base64array));

    }

    private class Networkconnection extends AsyncTask<Void, Void, String> {

        String result;
        String new_result;
        @Override
        protected String doInBackground(Void... params) {
            Response response;
            OkHttpClient client = new OkHttpClient();
            String byteValue = String.valueOf(base64array);
            //RequestBody body = RequestBody.create(MediaType.parse("json"), base64array);//RequestBody.create(MediaType.parse("application/json"),base64array);

            try {
                MediaType MEDIA_TYPE_MARKDOWN
                        = MediaType.parse("text/x-markdown; charset=utf-8");


                Request request = new Request.Builder()
                        .url(url)
                        .post(RequestBody.create(MEDIA_TYPE_MARKDOWN, byteValue))
                        .build();
                 response = client.newCall(request).execute();
                result = response.body().string();
            } catch (Exception e) {
                Log.d("Error: " , e.getMessage());
            }

            return result;
        }

        @Override
        protected void onPostExecute(String s) {
            Log.d("Anawe3r:" , result);
            String fina_answer = null;
            int index = result.indexOf(":") + 1;
            fina_answer = result.substring(index, result.length()-1);
            textcaptured.setText(fina_answer);


            //textcaptured.setText(result);
            super.onPostExecute(s);
        }
    }

}
