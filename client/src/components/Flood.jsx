import axios from 'axios';
import React, { useState } from 'react'

function Flood({ setPredicated ,setstate}) {
    const [floodData, setfloodData] = useState({ month: "jan", value: 0 })
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const handleMonthChange = (event) => {
        setfloodData({ ...floodData, month: event.target.value });
    };
    const handleClick = async (e) => {
        e.preventDefault();
        const res = await axios.post("http://127.0.0.1:5001/flood", floodData);
        console.log(res.data);
        const { predicted_FLOOD } = res.data;
        setPredicated(predicted_FLOOD);
        setstate(2);
    }
    return (
        <>
            <div className='flex flex-wrap flex-col w-full h-full items-center justify-center p-5 gap-4'>
                <div className='w-full flex items-center justify-center gap-7'>
                    <select value={floodData.month} onChange={handleMonthChange} className='p-2 px-4 rounded-sm text-black'>
                        {months.map((month) => (
                            <option key={month} value={month.toLowerCase()}>{month}</option>
                        ))}
                    </select>
                    <input
                        name='value'
                        placeholder='Enter the Value'
                        id='value'
                        onChange={(e) => setfloodData({ ...floodData, value: Number(e.target.value) })}
                        className='bg-white border-[1px] border-black text-black text-sm py-3 px-10 lg:px-7 flex items-center justify-center font-semibold rounded-sm transition duration-150 ease-in-out'
                    />
                </div>
                <div>
                    <button className='bg-white text-black p-3 px-8 flex items-center justify-center font-semibold rounded-md transition duration-150 ease-in-out hover:bg-gray-500 hover:text-white' onClick={handleClick}>Show data</button>
                </div>
            </div>
        </>
    )
}

export default Flood