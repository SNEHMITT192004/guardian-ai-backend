import axios from 'axios';
import React, { useState } from 'react'

function Cyclone({ setPredicated,setstate }) {

    const [hurriData, sethurriData] = useState({ latitude: 0, longitude: 0, moderate_wind_ne: 0, moderate_wind_se: 0, moderate_wind_sw: 0, moderate_wind_nw: 0, year: 0, month: 0, day: 0 })

    const handleClick = async (e) => {
        e.preventDefault();
        const res=await axios.post("http://127.0.0.1:5001/hurri",hurriData);
        console.log(hurriData)
        console.log(res.data);
        const {predicted_max_wind}=res.data;
        setPredicated(predicted_max_wind);
        setstate(3);
    }
    return (
        <>
            <div className='flex flex-col w-full h-full items-center justify-around p-5 lg:px-3 gap-4'>
                <div className='flex flex-wrap gap-4 items-center justify-around'>
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='latitude'
                        type='number'
                        placeholder='Latitude'
                        onChange={(e) => sethurriData({ ...hurriData, latitude: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='longitude'
                        type='number'
                        placeholder='Longitude'
                        onChange={(e) => sethurriData({ ...hurriData, longitude: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='moderate_wind_ne'
                        type='number'
                        placeholder='moderate_wind_ne'
                        onChange={(e) => sethurriData({ ...hurriData, moderate_wind_ne: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='moderate_wind_se'
                        type='number'
                        placeholder='moderate_wind_se'
                        onChange={(e) => sethurriData({ ...hurriData, moderate_wind_se: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='moderate_wind_sw'
                        type='number'
                        placeholder='moderate_wind_sw'
                        onChange={(e) => sethurriData({ ...hurriData, moderate_wind_sw: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='moderate_wind_nw'
                        type='number'
                        placeholder='moderate_wind_nw'
                        onChange={(e) => sethurriData({ ...hurriData, moderate_wind_nw: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='year'
                        type='number'
                        placeholder='year'
                        onChange={(e) => sethurriData({ ...hurriData, year: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='month'
                        type='number'
                        placeholder='month'
                        onChange={(e) => sethurriData({ ...hurriData, month: Number(e.target.value) })}
                    />
                    <input
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                        id='day'
                        type='number'
                        placeholder='day'
                        onChange={(e) => sethurriData({ ...hurriData, day: Number(e.target.value) })}
                    />
                </div>
                <div>
                    <button className='bg-white text-black p-3 px-8 flex items-center justify-center font-semibold rounded-md transition duration-150 ease-in-out hover:bg-gray-500 hover:text-white' onClick={handleClick}>Show data</button>
                </div>
            </div>
        </>
    )
}

export default Cyclone