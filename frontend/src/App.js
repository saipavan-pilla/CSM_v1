import React from 'react';
import './App.css';
import "react-responsive-carousel/lib/styles/carousel.min.css";
import { Carousel } from 'react-responsive-carousel';
import logo from './pumalogopng33-1@2x.png'; 
import m1 from './1.png'; 
import m2 from './m1.png'; 
import m3 from './m2.png'; 
import m4 from './m3.png'; 
import c1 from './surya2x.png';
import c2 from './pavan2x.png';
import c3 from './manoj2x.png';
import c4 from './mani2x.png';
import c5 from './sandeep2x.png';
import  { useState } from 'react';

function App() {

  const handleSubmit = (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);
    
    for (const entry of formData.entries()) {
      console.log(entry[0], entry[1]);
    }
  };
 
  const [empInfo, setEmpInfo] = useState({
    name: 'Surya Prakash',
    hired: 'Jan 2023',
    rating: '4.5',
    sale: '520'
  });
   

 
  const updateName = (name, hired, rating, sale) => {
    setEmpInfo({
      name: name,
      hired: hired,
      rating: rating,
      sale: sale
    });
  };
  
  const [videoSource, setVideoSource] = useState("https://drive.google.com/file/d/1zjpSmTW80e_abW--uS-lhHRNhoLqObtp/preview");
 

   function showVideo1() {
    setVideoSource("https://drive.google.com/file/d/13m4ZNfJ57wkE29yD9oWmA94jm_r0HN32/preview");
  }
  
  function showVideo2() {
    setVideoSource("https://drive.google.com/file/d/13dVdNWJFll1Y9aV82S6zuUkthqJCzEcD/preview");
  }
  
  function showVideo3() {
    setVideoSource("https://drive.google.com/file/d/13kMPrrDILxXL15q9lsmt8V6rJTSh4IHR/preview");
  }
  
  function showVideo4() {
    setVideoSource("https://www.youtube.com/embed/zn56t5jz43o");
  }
  

  return (
    <div className='body'>
      {/* <nav className="navbar">
        <div className="navbar-container">
          <img src={logo} alt="My Logo" className="navbar-logo" />
          <div className="mobile-menu-icon">&#9776;</div>
          <ul className="nav-menu">
            <li className="nav-item">
              <a href="#home">Home</a>
            </li>
            <li className="nav-item">
              <a href="#about">About</a>
            </li>
            <li className="nav-item">
              <a href="#services">Services</a>
            </li>
            <li className="nav-item">
              <a href="#contact">Contact</a>
            </li>
          </ul>
        </div>
      </nav> */}
      <nav id="" class=" navbar navbar-expand-lg   fixed-top">
            <img src={logo} alt="My Logo" className="navbar-logo" />
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
                <div class="navbar-nav ml-auto">
                <a class="nav-link active" id="navitem1" href="#sectionwcu">Home<span class="sr-only">(current)</span></a>
                    <a class="nav-link" id="navitem2" href="#ems">About</a>
                    <a class="nav-link" id="navitem3" href="#dp">Services</a>
                    <a class="nav-link" id="navitem4" href="#fu">Contact Us</a>
                </div>
            </div>
       
    </nav>
    
      <div className="carousel-wrapper">
      <Carousel autoPlay interval={3000} infiniteLoop showThumbs={false} showStatus={true}>
        <div>
          <img src={m1} alt="Slide 1" />
        </div>
        <div>
          <img src={m2} alt="Slide 2" />
        </div>
        <div>
          <img src={m3} alt="Slide 3" />
        </div>
        <div>
          <img src={m4} alt="Slide 4" />
        </div>
      </Carousel>
      </div>
      <div className='staff'>
      <h1 class="c1">Employees</h1>
      <div className="image-container">
      <div className="col-12 mb-3 col-md-2">
          <div className="c1">
              <img src={c1} onClick={() => updateName('Surya Kumar', 'Jan 2023', '4.5', '450')}  style={{ cursor: 'pointer' }}/>
              <h2 className='cname surya1' onClick={() => updateName('Surya Kumar', 'Jan 2023', '4.5', '450')}  style={{ cursor: 'pointer' }}>Surya</h2>
      </div>
      </div>
      <div className="col-12 mb-3 col-md-2">
          <div className="c1">
              <img onClick={() => updateName('Pavan', 'Feb 2022', '4.7', '570')}  style={{ cursor: 'pointer' }} src={c2}/>
              <h2 className='cname pavan' onClick={() => updateName('Pavan', 'Feb 2022', '4.7', '570')}  style={{ cursor: 'pointer' }}>Pavan</h2>
      </div>
      </div>
      <div className="col-12 mb-3 col-md-2">
          <div className="c1">
              <img src={c3}  onClick={() => updateName('Manoj', 'Dec 2022', '4.3', '620')}  style={{ cursor: 'pointer' }}/>
              <h2 className='cname manoj' onClick={() => updateName('Manoj', 'Dec 2022', '4.3', '620')}  style={{ cursor: 'pointer' }}>Manoj</h2>

      </div>
      </div>
      <div className="col-12 mb-3 col-md-2">
          <div className="c1">
              <img src={c4} onClick={() => updateName('Manikanta', 'Apr 2023', '4.1', '670')}  style={{ cursor: 'pointer' }} />
              <h2 className='cname manikanta' onClick={() => updateName('Manikanta', 'Apr 2023', '4.1', '670')}  style={{ cursor: 'pointer' }}>Manikanta</h2>
      </div>
      </div>
      <div className="col-12 mb-3 col-md-2">
          <div className="c1">
              <img src={c5} onClick={() => updateName('Sandeep', 'June 2022', '4.0', '430')}  style={{ cursor: 'pointer' }}/>
              <h2 className='cname sandeep' onClick={() => updateName('Sandeep', 'June 2022', '4.0', '430')}  style={{ cursor: 'pointer' }}>Sandeep</h2>
      </div>
      </div>

  </div>
   </div>
      <div className='detail'>
         <div>
         <span><b className='heading'>Name : </b></span>
         <span><b id="Empname">{empInfo.name}</b></span> 
         </div>
         <div >
         <span><b className='heading'>Hired : </b></span>
         <span><b id="Emphired">{empInfo.hired}</b></span> 
         </div>
         <div>
         <span><b className='heading'>Average-Rating : </b></span>
         <span><b id="Emprating">{empInfo.rating}</b></span> 
         </div>
         <div>
         <span><b className='heading'>Average-Sale : </b></span>
         <span><b id="Empsale">{empInfo.sale}</b></span> 
         </div>
      </div>
      <div class="container pt-5 pb-5">
       
            <div class="row">
                <div class="col-12 col-md-5">
                    <div className="links"  style={{ border: '2px solid #000', padding: '10px' }}>
                    <a className="link1" href="#" id="sv1" onClick={() => showVideo1()}><span><i className="fa fa-video-camera"></i></span>06-06-2023 10:00 AM ★★★★★</a>
                    <a className="link1" href="#" id="sv2" onClick={() => showVideo2()}><span><i className="fa fa-video-camera"></i></span>07-06-2023 09:30 AM ★★★★☆</a>
                    <a className="link1" href="#" id="sv3" onClick={() => showVideo3()}><span><i className="fa fa-video-camera"></i></span>08-06-2023 11:32 AM ★★★☆☆</a>
                    <a className="link1" href="#" id="sv4" onClick={() => showVideo3()}><span><i className="fa fa-video-camera"></i></span>09-06-2023 09:45 AM ★★★★★</a>
                    <a className="link1" href="#" id="sv5" onClick={() => showVideo3()}><span><i className="fa fa-video-camera"></i></span>10-06-2023 05:30 PM ★★★★☆</a>                    
                    </div>
                </div>
                <div class="info col-12 col-md-7">
                <iframe class="video" id="video"  src={videoSource} width="640" height="480" allow="autoplay"></iframe>
                <div className='videoinfo'>
                  <div class="col-md-3">
                  <h4>Made Sale in</h4>
                  <b style={{color: '#55286f'}}>30 Mins</b>
                  </div>
                  <div class="col-md-2">
                  <h3>Performance</h3>
                  <b style={{padding:'5px',color: '#55286f'}}><span>&#9733;</span><span>&#9733;</span><span>&#9733;</span><span>&#9733;</span></b>
                  </div>
                  <div class="col-md-2">
                  <form action="/upload_video" method="post" encType="multipart/form-data" onSubmit={handleSubmit}>
                  <input type="file" name="video_file" />

                  <input style={{ display:'none' }} type="text" id="emp_name" name="text_name" value={empInfo.name} required /><br />
                  <input style={{ display:'none' }} type="text" id="emp_rating" name="text_rating" value={empInfo.rating} required />
                  <input style={{ display:'none' }} type="text" id="emp_sale" name="text_sale" value={empInfo.sale} required />

                  <input className="sub" type="submit" value="Upload" />
                </form>
                  </div>
                </div>  
                </div>
            </div>
       
    </div>
    <div className="footer ">
    <div className="des mb-3 col-12 col-md-3">
      <img src={logo} alt="My Logo" className="footer-logo" /><br></br>
      <div className='add'>
      <p>Cloudkarya.inc, Flat no:402,</p>
      <p>Vinayagar Heights,Waltair Uplands</p>
      <p>Visakhapatnam, Andhra Pradesh,</p>
      <p>Pin: 530003, India</p>
      </div> 
    </div>
      <div class="pow mb-3 col-12 col-md-5">
         <p className='pow1'>Powered By</p>
         <p>CloudKarya</p> 
      </div>
      <div class="add1 col-12 mb-3  col-md-3">
        <p><span><i className="fa fa-envelope" ></i></span>support@cloudkarya.com</p>
        <p><span><i className="fa fa-phone" ></i></span>0891 663 7995</p>
        <p className='icons'><span><i className='fab fa-instagram'></i></span>
        <span><i className='fab fa-facebook-f'></i></span>
        <span><i className='fab fa-twitter'></i></span>
        <i class='fab fa-discord'></i></p>
        
      </div>
    </div>
    </div>
  );
}

export default App;
