import React, { useState } from 'react'
import Earthquake from './Earthquake';
import Flood from './Flood';
import Cyclone from './Cyclone';

function LeftBox({ setPredicated ,setstate}) {
    const [earthquake, setEarthquake] = useState(true);
    const [flood, setFlood] = useState(false);
    const [cyclone, setCyclone] = useState(false);

    const isEarthquake = () => {
        setEarthquake(true);
        setFlood(false);
        setCyclone(false);
    }

    const isFlood = () => {
        setEarthquake(false);
        setFlood(true);
        setCyclone(false);
    }

    const isCyclone = () => {
        setEarthquake(false);
        setFlood(false);
        setCyclone(true);
    }
    return (
        <div className='w-full lg:w-[95vw] h-full lg:h-auto m-5 lg:mr-5 lg:my-5 lg:ml-0 lg:mt-28 py-5 lg:py-3 px-5 flex flex-col items-center justify-between bg-bg rounded-sm'>
            <div className='flex flex-wrap bg-screen lg:w-full h-full rounded-sm mb-4'>
                {earthquake && <Earthquake setPredicated={setPredicated} setstate={setstate} />}
                {flood && <Flood setPredicated={setPredicated}  setstate={setstate}/>}
                {cyclone && <Cyclone setPredicated={setPredicated} setstate={setstate} />}
            </div>
            <div className='flex items-center justify-around w-full gap-1 gap-y-3 xl:gap-3 flex-wrap'>
                <button className='bg-white border-[1px] border-black text-black text-sm p-3 px-12 flex items-center justify-center font-semibold rounded-full transition duration-150 ease-in-out hover:bg-gray-500 hover:text-white' onClick={isEarthquake}>Earthquake</button>
                <button className='bg-white border-[1px] border-black text-black text-sm p-3 px-12 flex items-center justify-center font-semibold rounded-full transition duration-150 ease-in-out hover:bg-gray-500 hover:text-white' onClick={isFlood}>Flood</button>
                <button className='bg-white border-[1px] border-black text-black text-sm p-3 px-12 flex items-center justify-center font-semibold rounded-full transition duration-150 ease-in-out hover:bg-gray-500 hover:text-white' onClick={isCyclone}>Cyclone</button>
            </div>
        </div>
    )
}

export default LeftBox