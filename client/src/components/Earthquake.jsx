import React, { useState } from 'react'
import axios from 'axios'

function Earthquake({ setPredicated ,setstate}) {
    const [earthquakeData, setEarthquakeData] = useState({ latitude: 0, longitude: 0 })
    const handleClick = async (e) => {
        e.preventDefault();
        const res = await axios.post("http://127.0.0.1:5001/earth", earthquakeData);
        console.log(res.data);
        const { predicted_magnitude } = res.data;
        setPredicated(predicted_magnitude);
        setstate(1);
    }
    return (
        <>
            <div className='flex flex-wrap w-full h-full items-center justify-center p-5 gap-4'>
                <input
                    name='latitude'
                    placeholder='Enter Latitude'
                    id='latitude'
                    onChange={(e) => setEarthquakeData({ ...earthquakeData, latitude: Number(e.target.value) })}
                    className='bg-white border-[1px] border-black text-black text-sm py-3 px-10 lg:px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                />
                <input
                    name='longitude'
                    placeholder='Enter Longitude'
                    id='longitude'
                    onChange={(e) => setEarthquakeData({ ...earthquakeData, longitude: Number(e.target.value) })}
                    className='bg-white border-[1px] border-black text-black text-sm py-3 px-10 lg:px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                />
                <div>
                    <button className='bg-white text-black p-3 px-8 flex items-center justify-center font-semibold rounded-md transition duration-150 ease-in-out hover:bg-gray-500 hover:text-white' onClick={handleClick}>Show data</button>
                </div>
            </div>
        </>
    )
}

export default Earthquake