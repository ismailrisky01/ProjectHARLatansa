package com.example.myapplication

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.myapplication.ml.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.util.*
import kotlin.collections.ArrayList

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
    val mode = Modus()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        handler = Handler(Looper.getMainLooper())
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        onStartRecord()
    }
    // onStartRecord merupakan fungsi untuk memulai sensor
    fun onStartRecord() {
        machineLearning = MachineLearning(baseContext)

        //untuk meregristasi sensor accelero dan gyro
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
//    private fun onStopRecord() {
//
//        sensorManager.unregisterListener(this)
//
//    }

    //berfungsi sebagai listener nilai sensor yang keluar, lalu setelah mendapat nilai, nilai akan di simpan ke dalam list dan dilakukan prediksi oleh varibale result
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
        val result = machineLearning.predict(normalizeData(stack.getArray()))
        Log.d("Latansa",result.toString())
        val text = findViewById<TextView>(R.id.result)
        val textRealtime = findViewById<TextView>(R.id.resultRealtime)
        mode.insert(result.toString())
        textRealtime.text = result.toString()

        if (mode.getModus()!=""){
            text.text = mode.getModus()
            Log.d("Latansa Modus",mode.getModus().toString())

            mode.clear()
        }else{

        }

//        Log.d("Latansa","Bismillah")
    }

    override fun onAccuracyChanged(p0: Sensor?, p1: Int) {
    }


}

//Pembuatan model Vector
data class Vector(val x: Float, val y: Float, val z: Float)

//Pembuatan model Dataset
data class DataSet(var accelerometer: Vector, var gyroscope: Vector, val state: State)

//Model State, berisi fitur"
enum class State(val category: Int) { SIT(0), STAND(1), WALK(2), SLEEP(3), UPSTAIRS(4), DOWNSTAIRS(5), STAND2SIT(6), STAND2SLEEP(7), SIT2STAND(8), SIT2SLEEP(9), SLEEP2STAND(10), SLEEP2SIT(11), UNKNOWN(12), JUMP(13), TORIGHT(14), TOLEFT(15) }
//enum class State(val category: Int) { UNKNOWN(0), SIT(1), STAND(2), WALK(3), SLEEP(4) }

//Class untuk membuat tumpukan yang nanti akan dimasukan ke prediksi
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
fun normalizeData(data: Array<Array<Float>>): Array<Array<Float>> {
    val flattenedData = data.flatten()
    val min = flattenedData.minOrNull() ?: 0.0f // Change to 0.0f to ensure Float type
    val max = flattenedData.maxOrNull() ?: 1.0f // Change to 1.0f to ensure Float type

    return data.map { row ->
        row.map { (it - min) / (max - min) }.toTypedArray()
    }.toTypedArray()
}
class Modus{
    private val arrayList = ArrayList<String>()
    fun insert(string: String){
        arrayList.add(string)
    }
    fun getModus():String{
        if (arrayList.size==15){
           return modus()
        }else{
            return ""
        }
    }
    fun clear(){
        arrayList.clear()
    }
    fun modus(): String {

        var previous = arrayList[0]
        var modus = arrayList[0]
        var count = 1
        var maxCount = 1

        for (i in 1 until arrayList.size) {
            if (arrayList[i] == previous) {
                count++
            }
            else {
                if (count > maxCount) {
                    modus = arrayList[i - 1]
                    maxCount = count
                }
                previous = arrayList[i]
                count = 1
            }
        }

        if (count > maxCount) {
            return arrayList[arrayList.size - 1]
        } else {
            return modus
        }
    }


}

fun main() {
    val list= listOf("a","b","a","a","b","b","b","b","b","b","b")
    val modus = Modus()
    list.forEach {
        modus.insert(it)
    }
    println("Value: "+modus.modus())
}
//Merubah data array ke data buffer
fun createBuffer(dataset: Array<Array<Float>>): ByteBuffer {
    val buffer = ByteBuffer.allocateDirect(dataset.size * dataset[0].size * 4)
    for (data in dataset) for (value in data) buffer.putFloat(Random().nextFloat())
    return buffer
}


//Fungsi untuk melakukan pemanggilan Model dan prediksi
class MachineLearning(context: Context) {

    private val model = ConvertedModelBaru.newInstance(context)
    private val inputLayer = TensorBuffer.createFixedSize(intArrayOf(128, 7), DataType.FLOAT32)

    fun predict(dataset: Array<Array<Float>>): State? {
        inputLayer.loadBuffer(createBuffer(dataset), intArrayOf(128, 7))
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
            output.getFloatValue(15),

        )
        val value = list.indices.maxByOrNull { list[it] } ?: 0
        return State.values().find { it.category == value }
    }
}
