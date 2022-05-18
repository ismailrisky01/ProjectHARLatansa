package com.example.myapplication

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.example.myapplication.ml.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.util.*

class MainActivity : AppCompatActivity(),SensorEventListener {
    var state: State = State.STAND
    private lateinit var sensorManager: SensorManager
    private var delay = true
    private var handler: Handler? = null
    private val dataSets = arrayListOf<DataSet>()
    lateinit var machineLearning: MachineLearning

    var acc = Vector(0f, 0f, 0f)
    var gyro = Vector(0f, 0f, 0f)
    val stack = Stack()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        handler = Handler(Looper.getMainLooper())
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        onStartRecord()
    }
    fun onStartRecord() {
        machineLearning = MachineLearning(baseContext)

        sensorManager.registerListener(
            this,
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
            SensorManager.SENSOR_DELAY_NORMAL,
            SensorManager.SENSOR_DELAY_UI
        )
        sensorManager.registerListener(
            this,
            sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
            SensorManager.SENSOR_DELAY_NORMAL,
            SensorManager.SENSOR_DELAY_UI
        )
    }
    private fun onStopRecord() {

        sensorManager.unregisterListener(this)

    }
    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> acc =
                Vector(event.values[0], event.values[1], event.values[2])
            Sensor.TYPE_GYROSCOPE -> gyro =
                Vector(event.values[0], event.values[1], event.values[2])
        }
        val dataset = DataSet(acc, gyro, state)
        dataSets.add(dataset)
        stack.insert(dataset)
        val result = machineLearning.predict(stack.getArray())
     Log.d("Latansa",result.toString())
    }

    override fun onAccuracyChanged(p0: Sensor?, p1: Int) {
    }


}
data class Vector(val x: Float, val y: Float, val z: Float)

data class DataSet(var accelerometer: Vector, var gyroscope: Vector, val state: State)

//enum class State(val category: Int) { UNKNOWN(0), SIT(1), STAND(2), WALK(3), SLEEP(4) }
enum class State(val category: Int) { SIT(0), STAND(1), WALK(2), SLEEP(3), UPSTAIRS(4), DOWNSTAIRS(5), STAND2SIT(6), STAND2SLEEP(7), SIT2STAND(8), SIT2SLEEP(9), SLEEP2STAND(10), SLEEP2SIT(11), UNKNOWN(12), JUMP(13), TORIGHT(14), TOLEFT(15) }

class Stack {
    private val arraylist = ArrayList<Array<Float>>().apply {
        for (i in 0 until 128) add(arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 1f))
    }

    private fun convert(item:DataSet): Array<Float> {
        return arrayOf(
            item.accelerometer.x, item.accelerometer.y, item.accelerometer.z,
            item.gyroscope.x, item.gyroscope.y, item.gyroscope.z,
            item.state.category.toFloat()
        )
    }

    fun insert(item: DataSet) {
        if (arraylist.count() >= 128)
            arraylist.removeFirst()
        arraylist.add(convert(item))
    }

    fun getArray(): Array<Array<Float>> {
        return arraylist.toTypedArray()
    }
}


fun createBuffer(dataset: Array<Array<Float>>): ByteBuffer {
    val buffer = ByteBuffer.allocateDirect(dataset.size * dataset[0].size * 4)
    for (data in dataset) for (value in data) buffer.putFloat(Random().nextFloat())
    return buffer
}


class MachineLearning(context: Context) {
    private val model = ConvertedModelBaru.newInstance(context)
    private val inputLayer = TensorBuffer.createFixedSize(intArrayOf(128, 7), DataType.FLOAT32)

    fun predict(dataset: Array<Array<Float>>): State? {
        Log.d("dataset", dataset.size.toString())
        Log.d("dataset", dataset[0].size.toString())
        inputLayer.loadBuffer(createBuffer(dataset))

        val output = model.process(inputLayer).outputFeature0AsTensorBuffer
        val list = arrayOf(
            output.getFloatValue(0),
            output.getFloatValue(1),
            output.getFloatValue(2),
            output.getFloatValue(3),
            output.getFloatValue(4),
            output.getFloatValue(5),
            output.getFloatValue(6),
            output.getFloatValue(7),
            output.getFloatValue(8),
            output.getFloatValue(9),
            output.getFloatValue(10),
            output.getFloatValue(11),
            output.getFloatValue(12),
            output.getFloatValue(13),
            output.getFloatValue(14),
            output.getFloatValue(15)

        )
        val value = list.indices.maxByOrNull { list[it] } ?: 0
        return State.values().find { it.category == value }
    }
}
