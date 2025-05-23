import React from "react";
import styled from "styled-components";

// Styled Components
const StyledSection01 = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  justify-content: flex-start;
  align-items: flex-start;
  display: inline-flex;
`;

const StyledFrame1 = styled.div`
  flex: 1 1 0;
  align-self: stretch;
  padding-top: 55px;
  padding-bottom: 30px;
  padding-left: 16px;
  padding-right: 16px;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  display: inline-flex;
`;

const StyledFrame2 = styled.div`
  align-self: stretch;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
  gap: 47px;
  display: flex;
`;

const StyledVelkommenText = styled.span`
  color: black;
  font-size: 90px;
  font-family: "Shadows Into Light", cursive;
  font-weight: 400;
  line-height: 140px;
  word-wrap: break-word;
  text-align: center;
  align-self: stretch;
`;

const StyledQuoteText = styled.span`
  color: black;
  font-size: 17px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
  word-wrap: break-word;
  text-align: center;
  align-self: stretch;
`;

const StyledTaglineText = styled.span`
  color: black;
  font-size: 11px;
  font-family: "Poppins", sans-serif;
  font-weight: 300;
  line-height: 15.4px;
  letter-spacing: 1.1px;
  word-wrap: break-word;
  text-align: center;
  align-self: stretch;
`;

const StyledImage = styled.img`
  flex: 1 1 0;
  align-self: stretch;
`;

// Component
export const Section01 = ({ title, subtitle, tagline, imageUrl }) => {
  return (
    <StyledSection01>
      <StyledFrame1>
        <StyledVelkommenText>{title}</StyledVelkommenText>
        <StyledFrame2>
          <StyledQuoteText>{subtitle}</StyledQuoteText>
          <StyledTaglineText>{tagline}</StyledTaglineText>
        </StyledFrame2>
      </StyledFrame1>
      <StyledImage src={imageUrl} alt="Selection image" />
    </StyledSection01>
  );
};
