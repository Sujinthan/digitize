package com.example.suji.define;


import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;

import java.io.IOException;
@SuppressWarnings("deprecation")
public class CameraView implements SurfaceHolder.Callback, Camera.PreviewCallback {

    private SurfaceHolder surface_Holder;
    private static Camera main_Camera;
    boolean on;
    Camera.Parameters parameters;

    Camera.AutoFocusCallback autoFocusCallback = new Camera.AutoFocusCallback() {
        @Override
        public void onAutoFocus(boolean success, Camera camera) {

        }
    };
    private  FocusBoxView focusBox;

    public CameraView(SurfaceHolder surfaceHolder){

            this.surface_Holder = surfaceHolder;
    }

    public boolean isOn(){
        return on;
    }


    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        try{
            this.main_Camera = Camera.open();

            this.main_Camera.setPreviewDisplay(this.surface_Holder);
            this.main_Camera.setDisplayOrientation(90);
            this.parameters  =  this.main_Camera.getParameters();
            String CurrentFocus = this.parameters.getFocusMode();
            if(CurrentFocus != null){
                Log.d("Message", "Inside CurrentFocus");
                this.parameters.setFocusMode("auto");
            }
            this.main_Camera.setParameters(this.parameters);
            this.main_Camera.setPreviewCallback(this);
           // main_Camera.autoFocus(autoFocusCallback);
        }catch (Exception e){
            Log.d("Error", "Canmera error on surfaceCreated" + e.getMessage());
            this.main_Camera.release();
            this.main_Camera = null;
        }

    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        this.parameters  =  this.main_Camera.getParameters();
        boolean CurrentFocus = this.parameters.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        if(CurrentFocus){
            Log.d("Message", "Inside CurrentFocus");
            this.parameters.setFocusMode("auto");
        }

        this.main_Camera.setParameters(this.parameters);
        if(holder.getSurface()==null){
            return;
        }
        try{
            main_Camera.stopPreview();
        }catch (Exception e){

        }
        try{

           this.main_Camera.setPreviewDisplay(this.surface_Holder);
            this.main_Camera.setDisplayOrientation(90);
            this.main_Camera.startPreview();
        }catch (IOException e){
            Log.d("Error", "Camera error on surfaceChanged " + e.getMessage());
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        this.main_Camera.setPreviewCallback(null);
        this.main_Camera.stopPreview();
        this.main_Camera.release();
        this.main_Camera= null;
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {

    }

    public static void takeShot(Camera.ShutterCallback shutterCallback,
                                Camera.PictureCallback rawPictureCallback,
                                Camera.PictureCallback jpegPictureCallback){
        main_Camera.takePicture(shutterCallback, rawPictureCallback, jpegPictureCallback);
    }
}
